#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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

const size_t N = 256 * 8; //如果h_A[i]全是1，N不能太大
#define FLOAT4(value)  *(float4*)(&(value))
#define block_size 256

__device__  float block_reduce_sum(float val)
{
    __shared__ float sdata[32];

    int warpID = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;//0~31

    for (int offset = warpSize/2; offset > 0; offset >>= 1)  
          val += __shfl_down_sync(0xffffffff, val, offset);

    if (lane == 0) sdata[warpID] = val;//每个warp里第一个线程的寄存器保存这个warp内部求和的结果
    __syncthreads();

    if(warpID == 0)
    {
        val = (lane < blockDim.x / warpSize) ? sdata[lane] : 0.0f;
        for (int offset = warpSize/2; offset > 0; offset >>= 1)  
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float block_reduce_max(float val)
{
    __shared__ float sdata[32];
    int warpID = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;

    for (int offset = warpSize/2; offset > 0; offset >>= 1)  
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));

    if (lane == 0) sdata[warpID] = val;
    __syncthreads();

    if(warpID == 0){
        val = (lane < blockDim.x / warpSize) ? sdata[lane] : -FLT_MAX;
        for (int offset = warpSize/2; offset > 0; offset >>= 1)  
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));//求最大值的时候必须得是__shfl_xor_sync
    }
    return val;
}

//https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void softmax(float* x, float* y, float* total)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    float sum = (idx < N) ? expf(x[idx]) : 0.0f;

    __shared__ float sdata[32];

    int warpID = tid / warpSize;
    int lane = tid % warpSize;

    // warp内部求和
    for (int offset = warpSize/2; offset > 0; offset >>= 1)  
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) sdata[warpID] = sum;
    __syncthreads();

    sum = (lane < blockDim.x / warpSize) ? sdata[lane] : 0.0f;
    for (int offset = warpSize/2; offset > 0; offset >>= 1)  
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __syncthreads();

    //if(warpID == 0){
        if (tid == 0) atomicAdd(total, sum);
        // 需要注意的是，__threadfence()本身不是同步操作,
        //它不阻塞当前线程，它只确保对total的写入，在其它block中的线程看来，的确发生在对total的读取之前
        //https://stackoverflow.com/questions/5232689/what-is-the-purpose-of-the-threadfence-intrinsic-in-cuda#
        __threadfence();
        if (idx < N) y[idx] = expf(x[idx]) / (*total); 
    //}
}

__global__ void softmax_v2(float* x, float* y, float* total)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    float sum = (idx < N) ? expf(x[idx]) : 0.0f;

    sum = block_reduce_sum(sum);

    __syncthreads();

    int warpID = tid / warpSize;

    //if(warpID == 0){
        if (tid == 0) atomicAdd(total, sum);
        __threadfence();

        if (idx < N) y[idx] = expf(x[idx]) / (*total); 

    //}

}

__global__ void softmax_safe(float* x, float* y, float* total, float* total_max)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    float ori_val = (idx < N) ? x[idx] : (-FLT_MAX); //FLT_MIN
    float max_val = block_reduce_max(ori_val);

    atomicMax(total_max, max_val);
    __threadfence();

    float exp_val = (idx < N) ? expf(ori_val - *total_max) : 0.0f;

    float sum = block_reduce_sum(exp_val);

    __syncthreads();

    int warpID = tid / warpSize;

    if (tid == 0) atomicAdd(total, sum);
    __threadfence(); 
    if (idx < N) y[idx] = exp_val / (*total);
}

__global__ void softmax_v2_float4(float* x, float* y, float* total)
{
    const int tid = threadIdx.x;
    const int idx = (blockIdx.x * blockDim.x + tid) * 4; 

    float4 sum = FLOAT4(x[idx]);
    float4 tmp_exp;

    tmp_exp.x = (idx < N) ? expf(sum.x) : 0.0f;
    tmp_exp.y = (idx < N) ? expf(sum.y) : 0.0f;
    tmp_exp.z = (idx < N) ? expf(sum.z) : 0.0f;
    tmp_exp.w = (idx < N) ? expf(sum.w) : 0.0f;

    float exp_val = (tmp_exp.x + tmp_exp.y + tmp_exp.z + tmp_exp.w);

    exp_val = block_reduce_sum(exp_val);

    __syncthreads();

    int warpID = tid / warpSize;
    //if(warpID == 0){
        if (tid == 0) atomicAdd(total, exp_val);
        __threadfence();

        if (idx < N) 
        {
            float4 tmp_y;
            tmp_y.x = tmp_exp.x / (*total);
            tmp_y.y = tmp_exp.y / (*total);
            tmp_y.z = tmp_exp.z / (*total);
            tmp_y.w = tmp_exp.w / (*total);
            FLOAT4(y[idx]) = tmp_y; 
        }
    //}
}

int main(){

    float *h_A, *d_A;
    float *h_B, *d_B;
    float *h_sum, *d_sum;
    float *h_max, *d_max;

    h_A = (float*)malloc(sizeof(float) * N);
    h_B = (float*)malloc(sizeof(float) * N);
    h_sum = new float;
    h_max = new float;

    for (int i = 0; i < N; i++)
    {
        h_A[i] = i/(float)N;//注意一定要加float，不然h_A[i]全是0
    }

    checkCudaErrors(cudaMalloc(&d_A, N*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_B, N*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_sum, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_sum, 0, sizeof(float)));
    checkCudaErrors(cudaMemset(d_max, 0, sizeof(float)));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msec = 0;
    int iteration = 1;
    checkCudaErrors(cudaEventRecord(start));
    for(int i = 0; i < iteration; i++)
    {
        // softmax<<<CeilDiv((int)N, block_size), block_size>>>(d_A, d_B, d_sum);
        // softmax_v2<<<CeilDiv((int)N, block_size), block_size>>>(d_A, d_B, d_sum);//12.63%,22.91us
        // softmax_safe<<<CeilDiv((int)N, block_size), block_size>>>(d_A, d_B, d_sum, d_max);
        // softmax_v2_float4<<<CeilDiv((int)N, block_size), block_size/4>>>(d_A, d_B, d_sum);//21.66%,18.18us
        softmax_v2_float4<<<CeilDiv((int)N/4, block_size), block_size>>>(d_A, d_B, d_sum);//17.50%,17.15us
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
    printf("softmax takes %.3f msec\n", msec/iteration);

    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    float* ref = (float*)malloc(sizeof(float) * N);

    for(int i = 0; i < N; i++)
        sum += expf(h_A[i]);

    for(int i = 0; i < N; i++)
        ref[i] = expf(h_A[i]) / sum;

    for(int i = 0; i < N; i++)
    {
        double err = fabs(h_B[i] - ref[i]);

        if(err > 1.e-5 || isnan(h_B[i]))
        {
            printf("ref[%d]:%f, h_B[%d]:%f, cpu_sum :%f, h_sum :%f, h_max :%f\n", i, ref[i], i, h_B[i], sum, *h_sum, *h_max);
            printf("wrong answer!\n");
            break;
        }
    }

    return 0;
}