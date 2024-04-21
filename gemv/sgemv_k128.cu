#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#define FLOAT4(value) *(float4*)(&(value))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum)
{
    if (warpSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (warpSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8 ); // 0-8, 1-9, 2-10, etc.
    if (warpSize >= 8)  sum += __shfl_down_sync(0xffffffff, sum, 4 ); // 0-4, 1-5, 2-6, etc.
    if (warpSize >= 4)  sum += __shfl_down_sync(0xffffffff, sum, 2 ); // 0-2, 1-3, 4-6, 5-7, etc.
    if (warpSize >= 2)  sum += __shfl_down_sync(0xffffffff, sum, 1 ); // 0-1, 2-3, 4-5, etc.

    return sum;
}

__global__ void Sgemv_k128(float* A, float* x, float* y, const int M, const int K)
{
    const int warp_size = 32;
    int laneId = threadIdx.x % warp_size;
    int current_row = blockDim.y * blockIdx.x + threadIdx.y;

    if (current_row >= M) return;

    float res = 0.0f;
    int kIteration = (K / warp_size) / 4;
    if (kIteration == 0) kIteration = 1;

    #pragma unroll
    for (int i = 0; i < kIteration; i++)
    {
        int current_col = (i * warp_size + laneId);
        float4 tmp_A = reinterpret_cast<float4 *>(&A[current_row * K])[current_col];
        float4 tmp_x = reinterpret_cast<float4 *>(x)[current_col];
        res += tmp_A.x * tmp_x.x;
        res += tmp_A.y * tmp_x.y;
        res += tmp_A.z * tmp_x.z;
        res += tmp_A.w * tmp_x.w;
    }

    res = warpReduceSum<warp_size>(res);

    if (laneId == 0) y[current_row] = res;
}

int main(int argc, char** argv)
{
    size_t M = 1024;
    // size_t M = 16384;
    size_t K = 128;

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_x = sizeof(float) * K;
    size_t bytes_y = sizeof(float) * M;

    float* h_A  = (float*)malloc(bytes_A);
    float* h_x  = (float*)malloc(bytes_x);
    float* h_y  = (float*)malloc(bytes_y);
    float* h_y1 = (float*)malloc(bytes_y);

    float* d_A;
    float* d_x;
    float* d_y;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_x, bytes_x));
    checkCudaErrors(cudaMalloc(&d_y, bytes_y));

    double duration[2] = {0.0f, 0.0f};
    double GFLOPS[2]   = {0.0f, 0.0f};
    double GFLOPs      = 2.0 * M * 1 * K;

    // 生成A的数据
    for (int i = 0; i < M * K; i++)
    {
        h_A[i] = (float)i / K;
    }

    // 生成x的数据
    for (int i = 0; i < K; i++)
    {
        h_x[i] = 1;
    }
    
    memset(h_y, 0, M * sizeof(float));
    memset(h_y1, 0, M * sizeof(float));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float mseTotal = 0.0f;
    int nIter = 1000;

    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(start));

    for (int run = 0; run < nIter; run++)
    {
        dim3 dimGrid(M / 4);
        dim3 dimBlock(32, 4);
        Sgemv_k128<<<dimGrid, dimBlock>>>(d_A, d_x, d_y, M, K);
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&mseTotal, start, stop));
    checkCudaErrors(cudaMemcpy(h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    duration[0] = mseTotal / nIter;
    GFLOPS[0]   = (GFLOPs *1.0e-9f) / (duration[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        GFLOPS[0],
        duration[0],
        GFLOPs);
    
    // cublas
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float alpha = 1.0f;
    float beta  = 0.0f;
    checkCudaErrors(cudaMemcpy(d_y, h_y1, bytes_y, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        cublasSgemv(
            blas_handle, CUBLAS_OP_T,
            K, M, &alpha,
            d_A, K, d_x, 1, &beta, d_y, 1
        );
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&mseTotal, start, stop));

    checkCudaErrors(cudaMemcpy(h_y1, d_y, bytes_y, cudaMemcpyDeviceToHost));

    duration[1] = mseTotal / nIter;
    GFLOPS[1]   = (GFLOPs * 1.0e-9f) / (duration[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        GFLOPS[1],
        duration[1],
        GFLOPs);
    
    cublasDestroy(blas_handle);

    double eps = 1.e-6; // match zero
    bool correct = true;
    for (int i = 0; i < M; i++)
    {
        double abs_err = fabs(h_y[i] - h_y1[i]);
        double dot_length = M;
        double abs_val = fabs(h_y[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_y[i], h_y1[i], eps);
            correct = false;
            break;
        }
    }
    
    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("My Gemm to CuBlas implementation ratio = %f\n", GFLOPS[0] / GFLOPS[1]);

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    free(h_A);
    free(h_x);
    free(h_y);
}