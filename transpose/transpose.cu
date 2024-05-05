#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define FLOAT4(value)  *(float4*)(&(value))
#define block_size 256

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

__global__ void mat_transpose_kernel_v0(
        const float* idata, float* odata,
        int M, int N)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= M || y >= N)
        return;

    odata[x * M + y] = idata[y * N + x];
}

void mat_transpose_v0(
    const float* idata, float* odata,
    int M, int N)
{
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    mat_transpose_kernel_v0<<<grid, block>>>(idata, odata, M, N);
}

template<const int BLOCK_SZ>
__global__ void mat_transpose_kernel_v1(
    const float* idata, float* odata,
    int M, int N)
{
    const int bx = blockIdx.x,  by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ];

    int x = bx * BLOCK_SZ + tx;
    int y = by * BLOCK_SZ + ty;

    if(x >= N || y >= M)
        return;

    sdata[ty][tx] = idata[y * N + x];

    __syncthreads();

    x = by * BLOCK_SZ + tx;
    y = bx * BLOCK_SZ + ty;

    if(x >= M || y >= N)
        return;

    odata[y * M + x] = sdata[tx][ty];
}

void mat_transpose_v1(const float* idata, float* odata, int M, int N)
{
    const int BLOCK_SZ = 16;
    dim3 block(BLOCK_SZ, BLOCK_SZ);
    dim3 grid(CeilDiv(N, BLOCK_SZ), CeilDiv(M, BLOCK_SZ));
    mat_transpose_kernel_v1<BLOCK_SZ><<<grid, block>>>(idata, odata, M, N);
}

template<const int BLOCK_SZ>
__global__ void mat_transpose_kernel_v2(
    const float* idata, float* odata,
    int M, int N)
{
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ + 1];// padding

    int x = bx * BLOCK_SZ + tx;
    int y = by * BLOCK_SZ + ty;

    if (y >= M || x >= N)
        return;

    sdata[ty][tx] = idata[y * N + x];
    __syncthreads();

    x = by * BLOCK_SZ + tx;
    y = bx * BLOCK_SZ + ty;
    if (y >= N || x >= M)
        return;

    odata[y * M + x] = sdata[tx][ty];
}

void mat_transpose_v2(const float* idata, float* odata, int M, int N)
{
    const int BLOCK_SZ = 16;
    dim3 block(BLOCK_SZ, BLOCK_SZ);
    dim3 grid(CeilDiv(N, BLOCK_SZ), CeilDiv(M, BLOCK_SZ));
    mat_transpose_kernel_v2<BLOCK_SZ><<<grid, block>>>(idata, odata, M, N);
}

template<const int BLOCK_SZ, const int NUM_PER_THREAD>
__global__ void mat_transpose_kernel_v3(const float* idata, float* odata, int M, int N)
{
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ+1];

    int x = bx * BLOCK_SZ + tx;
    int y = by * BLOCK_SZ + ty;

    const int ROW_STRIDE = BLOCK_SZ / NUM_PER_THREAD;

    if (x < N)
    {
        #pragma unroll
        for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
            if (y + y_off < M) {
                sdata[ty + y_off][tx] = idata[(y + y_off) * N + x]; 
            }
        }
    }
    __syncthreads();

    x = by * BLOCK_SZ + tx;
    y = bx * BLOCK_SZ + ty;
    if (x < M)
    {
        for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE)
        {
            if (y + y_off < N)
            {
                odata[(y + y_off) * M + x] = sdata[tx][ty + y_off];
            }
        }
    }
}

void mat_transpose_v3(const float* idata, float* odata, int M, int N)
{
    const int BLOCK_SZ = 32;
    const int NUM_PER_THREAD = 4;
    dim3 block(BLOCK_SZ, BLOCK_SZ/NUM_PER_THREAD);
    dim3 grid(CeilDiv(N, BLOCK_SZ), CeilDiv(M, BLOCK_SZ));
    mat_transpose_kernel_v3<BLOCK_SZ, NUM_PER_THREAD><<<grid, block>>>(idata, odata, M, N);
}

int main(int argc, char** argv)
{
    const int M = 1024;
    const int N = 1024;
    const int mem_size = M * N * sizeof(float);

    float *h_idata = (float*)malloc(mem_size);
    float *h_tdata = (float*)malloc(mem_size);

    float *d_idata, *d_tdata;
    checkCudaErrors(cudaMalloc(&d_idata, mem_size));
    checkCudaErrors(cudaMalloc(&d_tdata, mem_size));

    // host
    for (int i = 0; i < M * N; i++)
    {
        h_idata[i] = i / 666;
    }

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    checkCudaErrors(cudaEventCreate(&startEvent));
    checkCudaErrors(cudaEventCreate(&stopEvent));

    float milliseconds = 0;
    int iteration = 10;
    for (int run = 0 ; run < iteration; run ++)
    {
        // mat_transpose_v0(d_idata, d_tdata, M, N);
        // mat_transpose_v1(d_idata, d_tdata, M, N);
        // mat_transpose_v2(d_idata, d_tdata, M, N);
        mat_transpose_v3(d_idata, d_tdata, M, N);
    }

    checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(startEvent));
    for (int run = 0 ; run < iteration; run ++)
    {
        mat_transpose_v0(d_idata, d_tdata, M, N);
    }
    checkCudaErrors(cudaEventRecord(stopEvent));
    checkCudaErrors(cudaEventSynchronize(stopEvent));
    checkCudaErrors(cudaEventSynchronize(stopEvent));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

    checkCudaErrors(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));

    printf("transpose add takes %.5f msec\n", milliseconds / iteration);

    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++)
    {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(h_idata[col * N + row] - h_tdata[row * M + col]);
        double dot_length = M;
        double abs_val = fabs(h_idata[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_idata[i], h_tdata[col * M + row], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");

    free(h_idata);
    free(h_tdata);
    cudaFree(d_idata);
    cudaFree(d_tdata);

    return 0;
}