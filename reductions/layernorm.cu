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

__device__ float block_reduce_sum(float val)
{
    int lane = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;

    // 一个block里最多32个warps，因为最多1024个线程
    __shared__ float sdata[32];
    unsigned mask = 0xffffffff;

    //每个warp内部求和
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(mask, val, offset);
    }

    // 每个warp里第一个线程的寄存器保存这个warp内部求和的结果
    if (lane == 0)
    {
        sdata[warpID] = val;
    }
    __syncthreads();

    if (warpID == 0)
    {
        val = (lane < blockDim.x / warpSize) ? sdata[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(mask, val, offset);
        }
    }

    return val;
}

// Layer Norm: x: NxK(K=256<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
__global__ void layer_norm(float* x, float* y, float g, float b, int row, int col)
{
    int tid = threadIdx.x; // 0..col-1
    int bid = blockIdx.x;  // 0..row-1
    int idx = blockDim.x * bid + tid;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;     // 一个block计算一行的均值，s_mean被这个block里所有线程共享
    __shared__ float s_variance; // 一个block计算一行的方差，s_variance被这个block里所有线程共享

    float value = (idx < row * col) ? x[idx] : 0.0f; // 只加载一次
    float sum = block_reduce_sum(value);

    if (tid == 0)
    {
       s_mean = sum / ((float)col + epsilon);
    }
    __syncthreads(); // 一个block里所有线程同步

    float variance = (value - s_mean) * (value - s_mean);
    variance = block_reduce_sum(variance);

    if (tid == 0)
    {
        s_variance = rsqrtf(variance / ((float)col + epsilon));
    }
    __syncthreads(); // 一个block里所有线程同步

    if (idx < row * col)
    {
        y[idx] = ((value - s_mean) * s_variance) * g + b;
    }
}

__global__ void layer_norm_float4_block(float* x, float* y, float g, float b, int row, int col)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = (bid * blockDim.x + tid) * 4;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;
    __shared__ float s_variance;

    float4 tmp_x = FLOAT4(x[idx]);
    float value = (idx < row * col) ? (tmp_x.x + tmp_x.y + tmp_x.z + tmp_x.w) : 0.0f;

    float sum = block_reduce_sum(value);
    if (tid == 0)
    {
        s_mean = sum / ((float)col + epsilon);
    }
    __syncthreads();

    float4 tmp_x_hat;
    tmp_x_hat.x = tmp_x.x - s_mean;
    tmp_x_hat.y = tmp_x.y - s_mean;
    tmp_x_hat.z = tmp_x.z - s_mean;
    tmp_x_hat.w = tmp_x.w - s_mean;
    float variance =
        tmp_x_hat.x * tmp_x_hat.x +
        tmp_x_hat.y * tmp_x_hat.y +
        tmp_x_hat.z * tmp_x_hat.z +
        tmp_x_hat.w * tmp_x_hat.w;
    variance = block_reduce_sum(variance);

    if (tid == 0)
    {
        s_variance = rsqrtf(variance / ((float)col + epsilon));
    }
    __syncthreads();

    float4 tmp_y;
    tmp_y.x = tmp_x_hat.x * s_variance * g + b;
    tmp_y.y = tmp_x_hat.y * s_variance * g + b;
    tmp_y.z = tmp_x_hat.z * s_variance * g + b;
    tmp_y.w = tmp_x_hat.w * s_variance * g + b;
    if (idx < row * col)
    {
        FLOAT4(y[idx]) = tmp_y;
    }
}

/*
256/4 = 64
0,  1,  2, .....,  K-1
1
2
.
.
.
,
N-1
*/

//4x128
/*
0   127
128  255
256  xx
xxx  zzz
*/

//1x128
//4 warps, 32 threads

//1x256, 256 col
//float4   4 elements per threads , 256 in total, 64 threads -> 2 warps
// 8 warp, 32 threads (0 warp + 1 warp)
__global__ void layer_norm_float4_grid(float* x, float* y, float g, float b, int row, int col)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = (bid * blockDim.x +tid) * 4;
    
    int lane = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;
    const float epsilon = 1e-5f;

    __shared__ float s_mean[32];
    __shared__ float s_variance[32];

    float4 tmp_x = FLOAT4(x[idx]);
    float value = (idx < row * col) ? (tmp_x.x + tmp_x.y + tmp_x.z + tmp_x.w) : 0.0f;

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0)
    {
        s_mean[warpID] = value / ((float)col + epsilon);
    }
    __syncthreads();

    float4 tmp_x_hat;
    tmp_x_hat.x = tmp_x.x - s_mean[warpID];
    tmp_x_hat.y = tmp_x.y - s_mean[warpID];
    tmp_x_hat.z = tmp_x.z - s_mean[warpID];
    tmp_x_hat.w = tmp_x.w - s_mean[warpID];
    float variance = 
            tmp_x_hat.x * tmp_x_hat.x +
            tmp_x_hat.y * tmp_x_hat.y +
            tmp_x_hat.z * tmp_x_hat.z +
            tmp_x_hat.w * tmp_x_hat.w;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        variance += __shfl_down_sync(0xffffffff, variance, offset);
    }
    if (lane == 0)
    {
        s_variance[warpID] = rsqrtf(variance / ((float)col + epsilon));
    }
    __syncthreads();

    float4 tmp_y;
    tmp_y.x = tmp_x_hat.x * s_variance[warpID] * g + b;
    tmp_y.y = tmp_x_hat.y * s_variance[warpID] * g + b;
    tmp_y.z = tmp_x_hat.z * s_variance[warpID] * g + b;
    tmp_y.w = tmp_x_hat.w * s_variance[warpID] * g + b;
    if (idx < row * col)
    {
        FLOAT4(y[idx]) = tmp_y;
    }
}

int main()
{
    size_t row = 1024;
    size_t N = row * block_size;
    size_t bytes_A = sizeof(float) * N;
    size_t bytes_B = sizeof(float) * N;

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);

    for( int i = 0; i < N; i++ ){
        h_A[i] = i;
    }

    float* d_A;
    float* d_B;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msec = 0;
    checkCudaErrors(cudaEventRecord(start));

    float g = 1.0f;
    float b = 1.0f;
    // layer_norm<<<CeilDiv((int)N, block_size),block_size>>>(d_A, d_B, g, b, row, block_size);
    // layer_norm_float4_block<<<CeilDiv((int)N, block_size),block_size/4>>>(d_A, d_B, g, b, row, block_size);
    layer_norm_float4_grid<<<CeilDiv((int)N/4, block_size),block_size>>>(d_A, d_B, g, b, row, block_size);

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
    checkCudaErrors(cudaMemcpy(h_B, d_B, bytes_B, cudaMemcpyDeviceToHost));

    printf("layernrom takes %.3f msec\n", msec);

    // Layer Norm: x: NxK(K=256), y': NxK, y'=x-mean(x)/std(x) each row
    // mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
    float* ref = (float*)malloc(sizeof(float) * N);

    //h_A(256 * 256)
    for(int i = 0; i < row; i++)
    {
        float sum = 0.0f;
        float mean = 0.0f;
        for(int j = 0; j < block_size; j++)
        {
            sum += h_A[i * block_size + j];
        }

        mean = sum /block_size;
        float std_sum = 0.0f;
        for(int j = 0; j < block_size; j++)
        {

            std_sum += (h_A[i * block_size + j] - mean) * (h_A[i * block_size + j] - mean);
        }
        std_sum /= block_size;

        float reverse = rsqrtf(std_sum);

        for(int j = 0; j < block_size; j++)
        {
            ref[i * block_size + j] = (h_A[i * block_size + j] - mean) * reverse * g + b;
        }
    }
    
    for(int i = 0; i < N; i++)
    {
        double err = fabs(h_B[i] - ref[i]);
        
        if(err > 1.e-2 || isnan(h_B[i]))
        {
            printf("i :%d, h_B[%d] :%lf, ref[%d] :%lf\n", i, i, h_B[i], i, ref[i]);
            printf("wrong answer!\n");
            break;
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);

    free(h_A);
    free(h_B);
    return 0;
}