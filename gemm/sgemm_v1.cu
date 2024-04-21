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

// SGEMM: Block Tile + Thread Tile + K Tile + Vec4, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
__global__ void sgemm_thread_tile_vec4(float* global_a, float* global_b, float* global_c, int M, int N, int K)
{
    /*
    每个c tile的大小是128X128，线程块block大小是16x16，
    线程块里的每个线程thread负责thread_m*thread_n(8*8)个元素，增加计算密度
    在K维度上继续进行分块，每块大小为block_k，迭代(K+block_k-1/block_k)次，
    每次迭代计算thread_m*thread_n个元素各自的部分乘累加(和上一次迭代的结果进行累加)
    利用float4进行向量化访存，一次读4个float类型的元素，128字节
    */
    constexpr int block_m = 128;
    constexpr int block_n = 128;
    constexpr int block_k = 8;
    constexpr int thread_m = 8;
    constexpr int thread_n = 8;
    // 2*128*8*4 = 8KB
    __shared__ float shared_a[block_m][block_k];
    __shared__ float shared_b[block_k][block_n];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 每个线程块内部的线程索引

    int sm_am = tid / 2;
    int sm_ak = (tid % 2 == 0) ? 0 : 4;
    /*
    针对线程块block里某一个编号是tid的线程，计算它负责的元素在a的shared memory里的索引
    shared_a的形状是128x8，128行，每行8个元素，每个线程需要读取4个元素，一行需要2个线程
    总共128行，需要的线程数刚好是2x128=256
    shared_a的大小是128x8,tid的编号是0~255，所以行索引sm_am直接用tid除以2就可以
    由于采用了float4向量化访存的技术，列索引sm_ak的取值只有0和4两种情况，需要看tid是否能被2整除
    */

    int sm_bk = tid / 32;
    int sm_bn = (tid % 32) * 4;
    /*
    针对线程块block里某一个编号是tid的线程，计算它负责的元素在b的shared memory里的索引
    shared_b的形状是8x128，8行，每行128个元素，每个线程需要读取4个元素，一行需要32个线程
    总共8行，需要的线程数刚好是8x32=256
    tid的编号是0~255，所以行索引sm_bk需要除以32
    由于采用了float4向量化访存的技术，列索引sm_bn的取值有0,4,8,...,124一共32种情况，计算方式是(tid % 32) * 4;
    */

    // 有了在shared_memory里的索引之后，就可以利用它进一步计算当前线程tid负责计算的元素在全局内存里的索引
    // 线程块索引为(bx,by)的block里每个thread负责计算C中大小为block_m*block_n(8x8)的块
    int gm_am = by * block_m + sm_am; // 当前线程负责加载元素在a矩阵(全局内存)中的行索引
    int gm_bn = bx * block_n + sm_bn; // 当前线程负责加载元素在b矩阵(全局内存)中的列索引

    float r_c[thread_m][thread_n] = {0.0f}; // 8x8个元素，这些元素是用寄存器保存的
    //大迭代，在K这个维度上进行分块，每块block_k大小，由于K是动态传入的变量，这个for循环无法展开
    for (int bk = 0; bk < (K + block_k - 1) / block_k; bk++)
    {
        // 共享内存shared_a[block_m=128][block_k=8]
        int gm_ak = bk * block_k + sm_ak; // 当前线程负责加载元素在a矩阵(全局内存)中的列索引
        int gm_a_offset = gm_am * K + gm_ak;//gm_am和循环变量bk没有关系，gm_ak才和循环变量bk有关
        FLOAT4(shared_a[sm_am][sm_ak]) = FLOAT4(global_a[gm_a_offset]);

        // 共享内存shared_b[block_k=8][block_n=128]
        int gm_bk = bk * block_k + sm_bk; // 当前线程负责加载元素在b矩阵(全局内存)中的行索引
        int gm_b_offset = gm_bk * N + gm_bn; //gm_bn和循环变量bk没有关系，gm_bk才和循环变量bk有关
        FLOAT4(shared_b[sm_bk][sm_bn]) = FLOAT4(global_b[gm_b_offset]); //采用float4向量化读取
        __syncthreads();

        // 由于block_k，thread_m，thread_n都是静态的编译期就能确定的变量，可以循环展开
        #pragma unroll
        for (int k = 0; k < block_k; k++)
        {
            // 小迭代，每个线程负责计算block_m*block_n(128x128)中的thread_m*thread_n(8x8)个元素
            #pragma unroll
            for (int m = 0; m < thread_m; m++)
            {
                #pragma unroll
                for (int n = 0; n < thread_n; n++)
                {
                    // k的取值范围是0~7，ty和tx的取值范围是0~15, 16x8=128
                    // 128*8 128/thread_m(8)=16 M方向 16线程
                    // 8*128 128/thread_n(8)=16 N方向 16线程
                    r_c[m][n] += shared_a[ty * thread_m + m][k] * shared_b[k][tx * thread_n + n];
                }
            }
        }
        __syncthreads();
    }

    // 由于thread_m，thread_n都是静态的编译期就能确定的变量，可以循环展开
    #pragma unroll
    for (int m = 0; m < thread_m; m++)
    {
        // 当前线程负责写入元素在c矩阵(全局内存)中的行索引
        int gm_cm = by * block_m + ty * thread_m + m;
        #pragma unroll
        for (int n = 0; n < thread_n; n += 4)
        {
            // 当前线程负责写入元素在c矩阵(全局内存)中的列索引
            int gm_cn = bx * block_n + tx * thread_n + n;
            // 采用float4向量化写入
            FLOAT4(global_c[gm_cm * N + gm_cn]) = FLOAT4(r_c[m][n]);
        }
    }
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
    double GFLOPS[2] = {0, 0};        //（全部大写）是floating-point operations per second的缩写，意指每秒浮点运算次数。用来衡量硬件的性能
    double GFLOPs = 2.0 * M * N * K;  //(s小写)是floating point of operations的缩写，是浮点运算次数，可以用来衡量算子/模型计算量

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;    

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
        sgemm_thread_tile_vec4<<<dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
    }

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < iteration; run ++ )
    {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        sgemm_thread_tile_vec4<<<dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&durationTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    duration[0] = durationTotal / iteration;
    GFLOPS[0] = (GFLOPs * 1.0e-9f) / (duration[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        GFLOPS[0],
        duration[0],
        GFLOPs);

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
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

    return 0;
}