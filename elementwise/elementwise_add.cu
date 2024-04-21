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

// ElementWise Add  
// elementwise_add<<<CeilDiv(N, block_size), block_size>>>(d_A, d_B, d_C, N);
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
//nvcc -o elementwise_add elementwise_add.cu && ./elementwise_add
//sudo /opt/nvidia/nsight-compute/2022.2.1/ncu --set full -f -o elementwise_add ./elementwise_add
__global__ void elementwise_add(float* a, float* b, float* c, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void elementwise_add_float4(float* a, float* b, float* c, int N)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;

    if (idx < N)
    {
        float4 tmp_a = FLOAT4(a[idx]);
        float4 tmp_b = FLOAT4(b[idx]);
        float4 tmp_c;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        tmp_c.z = tmp_a.z + tmp_b.z;
        tmp_c.w = tmp_a.w + tmp_b.w;
        FLOAT4(c[idx]) = tmp_c;
    }
}

int main()
{
    size_t block_size = 128;
    size_t N = 32 * 1024 * 1024;
    size_t bytes_A = sizeof(float) * N;
    size_t bytes_B = sizeof(float) * N;
    size_t bytes_C = sizeof(float) * N;

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);

    for (int i = 0; i < N; i++)
    {
        h_A[i] = i / 66;
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
    float msec = 0;

    int iteration = 1000;
    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < iteration; i++)
    {
        elementwise_add<<<CeilDiv(N, block_size), block_size>>>(d_A, d_B, d_C, N);
        // elementwise_add_float4<<<CeilDiv(N, block_size), block_size / 4>>>(d_A, d_B, d_C, N);
        // elementwise_add_float4<<<CeilDiv(N / 4, block_size), block_size>>>(d_A, d_B, d_C, N);
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
    printf("elementwise add takes %.5f msec\n", msec / iteration);

    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++)
    {
        double err = fabs(h_C[i] - (h_A[i] + h_B[i]));
        if (err > 1.e-6)
        {
            printf("Wrong answer!\n");
            break;
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}