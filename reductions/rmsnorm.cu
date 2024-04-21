#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define FLOAT4(value)  *(float4*)(&(value))
#define block_size 128
#define reduce_dim 256

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

// RMS Norm: x: NxK(K=256<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)  
__global__ void rms_norm(float* x, float* y, float g, float b, int row, int col)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const float epsilon = 1e-5f;

    __shared__ float s_variance;
    float value = (idx < row * col) ? x[idx] : 0.0f;
    float variance = value * value;
    variance = block_reduce_sum(variance);
    if (tid == 0)
    {
        s_variance = rsqrtf(variance / ((float)col + epsilon));
    }
    __syncthreads();

    if (idx < row * col)
    {
        y[idx] = (value * s_variance) * g;
    }
}

__global__ void rms_norm_float4_block(float* x, float* y, float g, float b, int row, int col)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = (bid * blockDim.x + tid) * 4;
    const float epsilon = 1e-5f;

    __shared__ float s_variance;
    
    float4 tmp_x = FLOAT4(x[idx]);
    float variance = (idx < row * col) ?
            (tmp_x.x * tmp_x.x +
             tmp_x.y * tmp_x.y +
             tmp_x.z * tmp_x.z +
             tmp_x.w * tmp_x.w) : 0.0f;

    variance = block_reduce_sum(variance);
    if (tid == 0)
    {
        s_variance = rsqrtf(variance / ((float)col + epsilon));
    }
    __syncthreads();

    float4 tmp_y;
    tmp_y.x = tmp_x.x * s_variance * g;
    tmp_y.y = tmp_x.y * s_variance * g;
    tmp_y.z = tmp_x.z * s_variance * g;
    tmp_y.w = tmp_x.w * s_variance * g;
    if(idx < row * col)
    {
        FLOAT4(y[idx]) = tmp_y;
    }
}

__global__ void rms_norm_float4_grid(float* x, float* y, float g, float b, int row, int col, float* row_mean)
{
    const int warp_size = 32;
    int laneId = threadIdx.x % warp_size;
    int current_row = blockDim.y * blockIdx.x + threadIdx.y;
    const float epsilon = 1e-5f;

    if (current_row < row)
    {
        float res = 0.0f;
        int kIteration = (col / warp_size) / 4;
        if (kIteration == 0)
            kIteration = 1;
        
        #pragma unroll
        for (int i = 0; i < kIteration; i++)
        {
            int current_col_vec = (i * warp_size + laneId);
            // float4 current_x= FLOAT4(x[current_row*K + current_col_vec]);//这么写会有misaligned address的报错
            float4 current_x = reinterpret_cast<float4 *>(&x[current_row * col])[current_col_vec];

            res += current_x.x * current_x.x;
            res += current_x.y * current_x.y;
            res += current_x.z * current_x.z;
            res += current_x.w * current_x.w;
        }

        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            res += __shfl_down_sync(0xffffffff, res, offset);
        if (laneId == 0)
            row_mean[current_row] = rsqrtf(res / ((float)col +epsilon));
        // __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < kIteration; i++)
        {
            int current_col_vec = (i * warp_size + laneId);
            float4 current_x = reinterpret_cast<float4 *>(&x[current_row * col])[current_col_vec];

            float4 current_y;
            current_y.x = current_x.x * row_mean[current_row];
            current_y.y = current_x.y * row_mean[current_row];
            current_y.z = current_x.z * row_mean[current_row];
            current_y.w = current_x.w * row_mean[current_row];

            reinterpret_cast<float4 *>(&y[current_row * col])[current_col_vec] = current_y;
        }
    }
}

int main()
{
    size_t row = 1024;
    size_t N = row * reduce_dim;
    size_t bytes_A = sizeof(float) * N;
    size_t bytes_B = sizeof(float) * N;
    size_t bytes_row_mean = sizeof(float) * row;

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_row_mean = (float*)malloc(row *sizeof(float));

    for( int i = 0; i < N; i++ )
    {
        h_A[i] = i;
    }

    float* d_A;
    float* d_B;
    float* d_row_mean;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_row_mean, row *sizeof(float)));

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row_mean, h_row_mean, bytes_row_mean, cudaMemcpyHostToDevice));

    dim3 dimGrid(row/4);
    dim3 dimBlock(32,4);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msec = 0;

    checkCudaErrors(cudaEventRecord(start));

    float g = 1.0f;
    float b = 0.0f;

    // rms_norm<<<CeilDiv((int)N, reduce_dim),reduce_dim>>>(d_A, d_B, g, b, row, reduce_dim);
    // rms_norm_float4_block<<<CeilDiv((int)N, reduce_dim), reduce_dim/4>>>(d_A, d_B, g, b, row, reduce_dim);
    
    // rms_norm_float4_grid<<<CeilDiv((int)N/4, reduce_dim),block_size>>>(d_A, d_B, g, b, row, reduce_dim, d_row_mean);
    rms_norm_float4_grid<<<dimGrid, dimBlock>>>(d_A, d_B, g, b, row, reduce_dim, d_row_mean);

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
    checkCudaErrors(cudaMemcpy(h_B, d_B, bytes_B, cudaMemcpyDeviceToHost));

    printf("rms_norm takes %.3f msec\n", msec);

    // RMS Norm: x: NxK(K=256<1024), y': NxK, y'=x/rms(x) each row
    // 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
    // grid(N), block(K<1024) N=batch_size*seq_len, K=hidden_size
    // y=y'*g (g: scale)  
    float* ref = (float*)malloc(sizeof(float) * N);

    //h_A(256 * 256)
    for(int i = 0; i < row; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < reduce_dim; j++)
        {
            sum += h_A[i * reduce_dim + j] * h_A[i * reduce_dim + j];
        }

        float reverse = rsqrtf(sum/reduce_dim);

        for(int j = 0; j < reduce_dim; j++)
        {
            ref[i * reduce_dim + j] = h_A[i * reduce_dim + j] * reverse * g + b;
        }
    }
    
    for(int i = 0; i < N; i++)
    {
        double err = fabs(h_B[i] - ref[i]);

        if(err > 1.e-4)
        {
            printf("i :%d, h_B[%d] :%lf, ref[%d] :%lf\n", i, i, h_B[i], i, ref[i]);
            printf("wrong answer!\n");
            break;
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_row_mean);

    free(h_A);
    free(h_B);
    free(h_row_mean);
    return 0;
}