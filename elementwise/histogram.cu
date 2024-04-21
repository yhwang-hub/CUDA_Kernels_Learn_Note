#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define INT4(value)    *(int4*)(&(value))
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

/*
x[i] = i
        -
      - -
    - - -
  - - - -
- - - - -
考虑一个warp里相邻线程对全局内存y的访问是否合并(coalesced global access)
warp thread[0]: 0, 1, 2, 3, 
warp thread[1]: 4, 5, 6, 7
*/
__global__ void histogram(int* x, int* y, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        atomicAdd(&y[x[idx]], 1);
    }
}

__global__ void histogram_int4(int* x, int* y, int N)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;

    if (idx < N)
    {
        int4 tmp_y = INT4(x[idx]);
        atomicAdd(&(y[tmp_y.x]), 1);
        atomicAdd(&(y[tmp_y.y]), 1);
        atomicAdd(&(y[tmp_y.z]), 1);
        atomicAdd(&(y[tmp_y.w]), 1);
    }
}

int main()
{
    size_t block_size = 128;
    size_t N = 32 * 1024 * 1024;
    size_t bytes_A = sizeof(int) * N;
    size_t bytes_B = sizeof(int) * N;

    int* h_A = (int*)malloc(bytes_A);
    int* h_B = (int*)malloc(bytes_B);

    for (int i = 0; i < N; i++)
    {
        h_A[i] = i;
    }
    
    int* d_A;
    int* d_B;

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
        // histogram<<<CeilDiv(N, block_size), block_size>>>(d_A, d_B, N);
        // histogram_int4<<<CeilDiv(N, block_size), block_size / 4>>>(d_A, d_B, N);
        histogram_int4<<<CeilDiv(N / 4, block_size), block_size>>>(d_A, d_B, N);
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
    printf("elementwise add takes %.5f msec\n", msec / iteration);

    checkCudaErrors(cudaMemcpy(h_B, d_B, bytes_B, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++)
    {
        double err = fabs(h_B[i] - iteration * 1.0f);
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