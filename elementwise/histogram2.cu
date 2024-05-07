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

#define DATA_LEN (100 * 1024 * 1024)

inline int rnd(float x)
{
    return static_cast<int>(x * rand() / RAND_MAX);
}

__global__ void cal_hist(unsigned char* buffer, unsigned int* hist, long data_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (index < data_size)
    {
        atomicAdd(&hist[buffer[index]],1);
        index += stride;
    }
}

__global__ void cal_hist2(unsigned char* buffer, unsigned int* hist, long data_size)
{
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (index < data_size)
    {
        atomicAdd(&temp[buffer[index]], 1);
        index += stride;
    }
    __syncthreads();

    atomicAdd(&(hist[threadIdx.x]), temp[threadIdx.x]);
}

int main(int argc, char** argv)
{
    unsigned char* buffer = new unsigned char[DATA_LEN];
    for (int i = 0; i < DATA_LEN; ++i)
    {
        buffer[i] = rnd(255);
        if (buffer[i] > 255)
        {
            printf("error\n");
        }
    }

    unsigned int* d_hist;
    checkCudaErrors(cudaMalloc((void**)&d_hist, sizeof(unsigned int) * 256));
    checkCudaErrors(cudaMemset(d_hist, 0, sizeof(int)));

    unsigned char* d_buffer;
    checkCudaErrors(cudaMalloc((void**)&d_buffer, sizeof(unsigned char) * DATA_LEN));
    checkCudaErrors(cudaMemcpy(d_buffer, buffer, sizeof(unsigned char) * DATA_LEN, cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    int block_num = prop.multiProcessorCount;

    cudaEvent_t start, end;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));
    checkCudaErrors(cudaEventRecord(start, 0));

    int block_size = 256;
    // cal_hist<<<block_num, 256>>>(d_buffer, d_hist, DATA_LEN); // 30.726688 ms
    // cal_hist2<<<block_num, 256>>>(d_buffer, d_hist, DATA_LEN);   // 4.517536  ms
    cal_hist2<<<CeilDiv(DATA_LEN, block_size), block_size>>>(d_buffer, d_hist, DATA_LEN);

    float elapsed_time;
    checkCudaErrors(cudaEventRecord(end, 0));
    checkCudaErrors(cudaEventSynchronize(end));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, end));
    printf("using time: %f ms\n", elapsed_time);

    unsigned int h_hist[256];
    checkCudaErrors(cudaMemcpy(h_hist, d_hist, sizeof(unsigned int) * 256, cudaMemcpyDeviceToHost));

    long hist_count = 0;
    for (int i = 0; i <256; ++i)
    {
        hist_count += h_hist[i];
    }
    printf("histogram sum: %d\n", hist_count);

    for (int i = 0; i < DATA_LEN; ++i)
    {
        h_hist[buffer[i]]--;
    }
    for (int i = 0; i < 256; ++i)
    {
        if (h_hist[i] != 0)
        {
            printf("cal error\n");
        }
    }
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(end));
    checkCudaErrors(cudaFree(d_hist));
    checkCudaErrors(cudaFree(d_buffer));

    delete[] buffer;

    return 0;

}