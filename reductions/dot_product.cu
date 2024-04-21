#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define FLOAT4(value)  *(float4*)(&(value))
#define warpSize 32

#define checkCudaErrors(func)               \
{                                   \
    cudaError_t e = (func);         \
    if(e != cudaSuccess)                                        \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));        \
}

__global__ void dot_product(float* a, float* b, float* c, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    int lane = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;

    __shared__ float sdata[32];
    float val = 0.0f;
    unsigned mask = 0xFFFFFFFFU;

    while (idx < N)
    {
        val += a[idx] * b[idx];
        idx += gridDim.x * blockDim.x;
    }
    __syncthreads();

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (lane == 0)
    {
        sdata[warpID] = val;
    }
    __syncthreads();

    if (warpID == 0)
    {
        val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(mask, val, offset);
        }

        if (tid == 0)
        {
            atomicAdd(c, val);
        }
    }
}

__global__ void dot_product_float4(float* a, float* b, float* c, int N)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    int tid = threadIdx.x;

    int lane = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;

    __shared__ float sdata[32];
    float val = 0.0f;
    unsigned mask = 0xffffffff;

    while (idx < N)
    {
        float4 tmp_a = FLOAT4(a[idx]);
        float4 tmp_b = FLOAT4(b[idx]);
        val += tmp_a.x * tmp_b.x;
        val += tmp_a.y * tmp_b.y;
        val += tmp_a.z * tmp_b.z;
        val += tmp_a.w * tmp_b.w;
        idx += (gridDim.x * blockDim.x) * 4;
    }
    __syncthreads();

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (lane == 0)
    {
        sdata[warpID] = val;
    }
    __syncthreads();
    
    if(warpID == 0)
    {
        val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(mask, val, offset);
        }
        
        if (tid == 0)
        {
            atomicAdd(c, val);
        }
    }
}

template <typename T>
inline T CeilDiv(const T& a, const T& b)
{
    return (a + b - 1) / b;
}

int main()
{
    size_t block_size = 256;
    size_t N = 8 * 1024; //太大了结果可能会错
    size_t bytes_A = sizeof(float) * N;
    size_t bytes_B = sizeof(float) * N;
    size_t bytes_C = sizeof(float);

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);

    for (int i = 0; i < N; i++)
    {
        h_A[i] = i / 666;
    }
    for (int i = 0; i < N; i++)
    {
        h_B[i] = i % 666;
    }

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));

    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    float msec = 0.0f;
    int iteration = 1; //太大了结果可能会错

    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < iteration; i++)
    {
        // dot_product<<<CeilDiv(N, block_size), block_size>>>(d_A, d_B, d_C, N);
        // dot_product_float4<<<CeilDiv(N, block_size), block_size / 4>>>(d_A, d_B, d_C, N);
        dot_product_float4<<<CeilDiv(N / 4, block_size), block_size>>>(d_A, d_B, d_C, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
    printf("dot_product takes %.3f msec\n", msec / iteration);

    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    float h_res = 0.0f;
    for (int i = 0; i < N; i++)
    {
        h_res += h_A[i] * h_B[i];
    }

    if ((*h_C / iteration) != h_res)
    {
        printf("*h_C/iteration :%f, h_res :%f\n", *h_C / iteration, h_res);
        printf("wrong answer!\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}