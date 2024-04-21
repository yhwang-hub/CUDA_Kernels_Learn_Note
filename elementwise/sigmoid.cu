#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define FLOAT4(value)  *(float4*)(&(value))

#define checkCudaErrors(func)               \
{                                   \
    cudaError_t e = (func);         \
    if(e != cudaSuccess)                                        \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));        \
}

template <typename T>
inline T CeilDiv(const T& a, const T& b)
{
    return (a + b - 1) / b;
}

//nvcc -o sigmoid sigmoid.cu && ./sigmoid
//sigmoid<<<CeilDiv(N, block_size), block_size>>>(d_A, d_B, N)
//a: Nx1, b: Nx1, c: Nx1, c = sigmoid(a, b)
__global__ void sigmoid(float* a, float* b, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        b[idx] = 1.0f / (1.0f + expf(-a[idx]));
    }
}

__global__ void sigmoid_float4(float* a, float* b, int N)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;

    if (idx < N)
    {
        float4 tmp_a = FLOAT4(a[idx]);
        float4 tmp_b;
        tmp_b.x = 1.0f / (1.0f + expf(-tmp_a.x));
        tmp_b.y = 1.0f / (1.0f + expf(-tmp_a.y));
        tmp_b.z = 1.0f / (1.0f + expf(-tmp_a.z));
        tmp_b.w = 1.0f / (1.0f + expf(-tmp_a.w));
        FLOAT4(b[idx]) = tmp_b;
    }
}

int main()
{
    size_t block_size = 128;
    size_t N = 1 * 1024;
    size_t bytes_A = sizeof(float) * N;
    size_t bytes_B = sizeof(float) * N;

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);

    for (int i = 0; i < N; i++)
    {
        h_A[i] = (i / 66) * ((i % 2 == 0) ? 1 : -1);
    }
    
    float* d_A;
    float* d_B;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));

    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msec = 0;

    int iteration = 1000;
    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < iteration; i++)
    {
        // sigmoid<<<CeilDiv(N, block_size), block_size>>>(d_A, d_B, N);
        // sigmoid_float4<<<CeilDiv(N, block_size), block_size / 4>>>(d_A, d_B, N);
        sigmoid_float4<<<CeilDiv(N / 4, block_size), block_size>>>(d_A, d_B, N);
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
    printf("elementwise add takes %.5f msec\n", msec / iteration);

    checkCudaErrors(cudaMemcpy(h_B, d_B, bytes_B, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++)
    {
        double err = fabs(h_B[i] - 1.0f/(1.0f + expf(-h_A[i])));
        if (err > 1.e-6)
        {
            printf("Wrong answer!\n");
            break;
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);

    free(h_A);
    free(h_B);

    return 0;
}