#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define FLOAT4(value)  *(float4*)(&(value))
#define INT4(value)    *(int4*)(&(value))
#define WARP_SIZE 32

#define checkCudaErrors(func)               \
{                                   \
    cudaError_t e = (func);         \
    if(e != cudaSuccess)                                        \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));        \
}

__global__ void sgemm(float* global_a, float* global_b, float* global_c, int M, int N, int K)
{
    // [1] Block Tile: 32x32的block处理c上一块32x32的元素计算
    // [2]     K Tile: 使用共享内存，并将K分块为BK大小的块
    /*
    考虑一种比较简单的情况:
    线程块block为32x32，这个block负责处理的元素也是32x32个
    共享内存也是开的32x32个元素
    */
    constexpr int block_m = 32;
    constexpr int block_n = 32;
    constexpr int block_k = 32;
    __shared__ float shared_a[block_m][block_k];
    __shared__ float shared_b[block_k][block_n];

    int bx = blockIdx.x;  // N方向的线程块索引
    int by = blockIdx.y;  // M方向的线程块索引
    int tx = threadIdx.x; // N方向，每个线程块内部线程的索引
    int tid = threadIdx.y * blockDim.x + tx; // 一个block内部的线程索引

    /*
    block大小是32x32，这32x32个线程一起把数据从global memory(a和b矩阵)搬运到shared memory
    里相应的位置处，读取元素个数是32x32x2(a和b各一个)。sizeof(float)=4, 总数据量为32x32x4x2 = 8KB
    每个线程从a和b中各读取一个元素
    */
    int sm_am = tid / 32; // a shared_memory的行索引
    int sm_ak = tid % 32; // a shared_memory的列索引
    int sm_bk = tid / 32; // b shared_memory的行索引
    int sm_bn = tid % 32; // b shared_memory的列索引

    int gm_am = by * block_m + sm_am; // 当前线程负责加载元素在a和c矩阵中的行索引
    int gm_bn = bx * block_n + sm_bn; // 当前线程负责加载元素在b和c矩阵中的列索引
    if (gm_am >= M || gm_bn >= N)
    {
        return;
    }

    float sum = 0.0f;
    // 大迭代，负责a中一行/b中一列的结果
    for (int bk = 0; bk < (K + block_k - 1) / block_k; bk++)
    {
        // MxK矩阵，第gm_am行，第bk * block_k + sm_ak列
        int a_offset = gm_am * K + bk * block_k + sm_ak;
        shared_a[sm_am][sm_ak] = global_a[a_offset]; // 32x32
        // KxN矩阵，第bk * block_k + sm_bk行，第gm_bn列
        int b_offset = (bk * block_k + sm_bk) * N + gm_bn;
        shared_b[sm_bk][sm_bn] = global_b[b_offset]; //32x32
        __syncthreads();

        // 小迭代，负责BK内部a的一行/b的一列的求和
        #pragma unroll
        for (int k = 0; k < block_k; k++)
        {
            // 操作的数据都在shared memory当中
            sum += shared_a[sm_am][k] * shared_b[k][sm_bn];
        }
        __syncthreads();
    }

    // 把结果sum写回c中索引是load_gmem_a_m * N + load_gmem_b_n的地方
    global_c[gm_am * N + gm_bn] = sum;
}

int main(int argc, char** argv)
{
    size_t M = 4096;
    size_t K = 4096;
    size_t N = 4096;

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));

    double duration[2] = {0, 0};
    double GFLOPS[2]   = {0, 0};        //（全部大写）是floating-point operations per second的缩写，意指每秒浮点运算次数。用来衡量硬件的性能
    double GFLOPs      = 2.0 * M * N * K;  //(s小写)是floating point of operations的缩写，是浮点运算次数，可以用来衡量算子/模型计算量

    const int BLOCK_SIZE_M  = 32;
    const int BLOCK_SIZE_N  = 32;
    const int THREAD_SIZE_X = 1;
    const int THREAD_SIZE_Y = 1;    

    for( int i = 0; i < M * K; i++ )
    {
        h_A[i] = i / 666;
    }

    for( int i = 0; i < K * N; i++ )
    {
        h_B[i] = i % 666;
    }

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float durationTotal = 0;
    int iteration = 10;

    for (int run = 0 ; run < iteration; run ++ )
    {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        sgemm<<<dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
    }

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < iteration; run ++ )
    {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        sgemm<<<dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&durationTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    duration[0] = durationTotal / iteration;
    GFLOPS[0]   = (GFLOPs * 1.0e-9f) / (duration[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        GFLOPS[0],
        duration[0],
        GFLOPs);

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta  = 0;
    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < iteration; run ++ )
    {
        cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, 
            d_A, K, d_B, N, &beta, d_C, N
        );
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&durationTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    duration[1] = durationTotal / iteration;
    GFLOPS[1] = (GFLOPs * 1.0e-9f) / (duration[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        GFLOPS[1],
        duration[1],
        GFLOPs);

    cublasDestroy(blas_handle); 
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++)
    {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(h_C[i] - h_C1[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_C[i], h_C1[col * M + row], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("My Gemm to CuBlas implementation ratio = %f\n", GFLOPS[0] / GFLOPS[1]);
    
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
}