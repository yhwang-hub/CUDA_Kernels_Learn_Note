#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(value)  *(float4*)(&(value))
#define INT4(value)    *(int4*)(&(value))
#define WARP_SIZE 32

#define checkCudaErrors(func)               \
{                                   \
    cudaError_t e = (func);         \
    if(e != cudaSuccess)                                        \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));        \
}

__global__ void sgemm_register_tile_vec4(
                    float* __restrict__ global_a, float* __restrict__ global_b, float* __restrict__ global_c,
                    int M, int N, int K)
{
    constexpr int block_m  = 128;
    constexpr int block_n  = 128;
    constexpr int block_k  = 8;
    constexpr int thread_m = 8;
    constexpr int thread_n = 8;
    __shared__ float shared_a[block_k][block_m];
    __shared__ float shared_b[block_k][block_n];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[thread_m];
    float r_comp_b[thread_n];
    float r_c[thread_m][thread_n] = {0.0f};

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = ty * blockDim.x + tx;

    int sm_am = tid / 2;
    int sm_ak = (tid % 2 == 0) ? 0 : 4;
    int sm_bk = tid / 32;
    int sm_bn = (tid % 32) * 4;

    int gm_am = by * block_m + sm_am;
    int gm_bn = bx * block_n + sm_bn;
    if (gm_am >= M || gm_bn >= N)
    {
        return;
    }

    for (int bk = 0; bk < (K + block_k - 1) / block_k; bk++)
    {
        int gm_ak = bk * block_k + sm_ak;
        int gm_a_offset = gm_am * K + gm_ak;
        int gm_bk = bk * block_k + sm_bk;
        int gm_b_offset = gm_bk * N + gm_bn;
        FLOAT4(r_load_a[0]) = FLOAT4(global_a[gm_a_offset]);
        FLOAT4(r_load_b[0]) = FLOAT4(global_b[gm_b_offset]);

        shared_a[sm_ak    ][sm_am] = r_load_a[0];
        shared_a[sm_ak + 1][sm_am] = r_load_a[1];
        shared_a[sm_ak + 2][sm_am] = r_load_a[2];
        shared_a[sm_ak + 3][sm_am] = r_load_a[3];
        FLOAT4(shared_b[sm_bk][sm_bn]) = FLOAT4(r_load_b[0]);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < block_k; k++)
        {
            FLOAT4(r_comp_a[0]) = FLOAT4(shared_a[k][ty * thread_m / 2              ]);
            FLOAT4(r_comp_a[4]) = FLOAT4(shared_a[k][ty * thread_m / 2 + block_m / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(shared_b[k][tx * thread_n / 2              ]);
            FLOAT4(r_comp_b[4]) = FLOAT4(shared_b[k][tx * thread_n / 2 + block_n / 2]);

            #pragma unroll
            for (int m = 0; m < thread_m; m++)
            {
                #pragma unroll
                for (int n = 0; n < thread_n; n++)
                {
                    r_c[m][n] += r_comp_a[m] * r_comp_b[n];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < thread_m / 2; i++)
    {
        int gm_cm = by * block_m + ty * thread_m / 2 + i;
        int gm_cn = bx * block_n + tx * thread_n / 2;
        int gm_c_offset = gm_cm * N + gm_cn;
        FLOAT4(global_c[gm_c_offset]) = FLOAT4(r_c[i][0]);
        FLOAT4(global_c[gm_c_offset + block_n / 2]) = FLOAT4(r_c[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < thread_m / 2; i++)
    {
        int gm_cm = by * block_m + block_m / 2 + ty * thread_m / 2 + i;
        int gm_cn = bx * block_n               + tx * thread_n / 2;
        int gm_c_offset = gm_cm * N + gm_cn;
        FLOAT4(global_c[gm_c_offset]) = FLOAT4(r_c[i + thread_m / 2][0]);
        FLOAT4(global_c[gm_c_offset + block_n / 2]) = FLOAT4(r_c[i + thread_m / 2][4]);
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
        sgemm_register_tile_vec4<<<dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);

    }

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < iteration; run ++ )
    {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        sgemm_register_tile_vec4<<<dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);

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