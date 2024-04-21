#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define FLOAT4(value)  *(float4*)(&(value))

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// const size_t N = 32ULL*1024ULL*1024ULL;
// const size_t N = 8ULL * 1024ULL * 1024ULL;
const size_t N = 640 * 256; // data size, 163840
const int BLOCK_SIZE = 256;

__global__ void atomic_red(float* gdata, float* out)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N)
    {
        atomicAdd(out, gdata[idx]);
    }
}

//sweep style parallel reduction,规约求和，使用shared memory
__global__ void reduce_a(float* gdata, float* out)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float sdata[BLOCK_SIZE];
    // extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;

    sdata[tid] += gdata[idx];
    __syncthreads();

    /* 该方法不存在back conflict*/
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* 该方法存在bank conflict
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if(tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        or

        int index = 2 * s * tid;
        if(index < blockDim.x)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    */

    if (tid == 0)
    {
        atomicAdd(out, sdata[0]);
    }
}

// 在一个block里面求和，先每个warp内部求和，结果保存在每个warp第一个线程的寄存器val里
// 然后将val暂存在共享内存sdata里
// 最后在第一个warp里，把val从共享内存里再读取出来，对warp之间再求和
__global__ void reduce_ws(float* gdata, float* out)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    int laneID = threadIdx.x % warpSize; // 0~31
    int warpID = threadIdx.x / warpSize;

    // 最大是32，代表warp的数目。因为一个block最多1024个线程，对应32个warp
    __shared__ float sdata[32];
    float val = 0.0f;

    // grid stride loop to load 
    while (idx < N)
    {
        val += gdata[idx];
        idx += gridDim.x * blockDim.x;
    }

    // 1st warp-shuffle reduction
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    // warp内部求和, 每个线程都把warp内部求和的结果放在自己本地的val变量上，但是只有lane==0的val被保存进shared_memory[warpID]里

    if (laneID == 0)
    {
        sdata[warpID] = val;
    }
    __syncthreads();

    if (warpID == 0)
    {
        // reload val from shared mem if warp existed
        val = (tid < blockDim.x / warpSize) ? sdata[laneID] : 0;//为什么用lane而不是warpID去索引？ 因为warpID是0才会进来
        // tid 小于8，那么lane也是0~7之间，可以用来索引sdata
        // final warp-shuffle reduction
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (tid == 0)
        {
            atomicAdd(out, val);
        }
    }
}

__global__ void reduce_ws_float4(float* gdata, float* out)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    int tid = threadIdx.x;

    int lane = threadIdx.x % warpSize; // 0~31
    int warpID = threadIdx.x / warpSize;

    __shared__ float sdata[32];
    float val = 0.0f;

    // grid stride loop to load 
    while (idx < N)
    {
        float4 tmp_input = FLOAT4(gdata[idx]);
        val += tmp_input.x;
        val += tmp_input.y;
        val += tmp_input.z;
        val += tmp_input.w;
        idx += (gridDim.x * blockDim.x) * 4;
    }

    // 1st warp-shuffle reduction
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    // warp内部求和, 每个线程都把warp内部求和的结果放在自己本地的val变量上，但是只有lane==0的val被保存进shared_memory[warpID]里

    if (lane == 0)
    {
        sdata[warpID] = val;
    }
    __syncthreads();
    
    if (warpID == 0)
    {
        // reload val from shared mem if warp existed
        val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0;//为什么用lane而不是warpID去索引？ 因为warpID是0才会进来
        // tid 小于8，那么lane也是0~7之间，可以用来索引sdata
        // final warp-shuffle reduction
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (tid == 0)
        {
            atomicAdd(out, val);
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
    float* h_A;
    float* h_sum;
    float* d_A;
    float* d_sum;

    h_A = new float[N]; // allocate space for data in host memory
    h_sum = new float;

    for (int i = 0; i < N; i++) // initialize matrix in host memory
    {
        h_A[i] = 1.0f;
    }

    cudaMalloc(&d_A, sizeof(float) * N);
    cudaMalloc(&d_sum, sizeof(float));
    cudaCheckErrors("cudaMalloc failer!");
    
    // copy matrix A to device:
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheckErrors("cudaMemset failure");

    // cuda processing sequence step 1 is complete
    atomic_red<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
    cudaCheckErrors("atomic reduction kernel launch failure");
    // cuda processing sequence step 2 is complete
    // copy vector sums from device to host:
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("atomic reduction kernel execution failure or cudaMemcpy H2D failure");
    // cuda processing sequence step 3 is complete
    if(*h_sum != (float)N)
    {
        printf("atomic sum reduction incorrect!\n");/*return -1;*/
    }
    else
    {
        printf("atomic sum reduction correct!\n");
    }

    const int blocks = 640;

    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheckErrors("cudaMemset failure");
    // cuda processing sequence step 1 is complete
    reduce_a<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
    cudaCheckErrors("reduction w/atomic kernel launch failure");
    // cuda processing sequence step 2 is complete
    // copy vector sums from device to host:
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("reduction w/atomic kernel execution failure or cudaMemcpy H2D failure");
    //cuda processing sequence step 3 is complete
    if(*h_sum != (float)N)
    {
        printf("reduction w/atomic sum incorrect!\n");
    }
    else
    {
        printf("reduction w/atomic sum correct!\n");
    }

    // cuda processing sequence step 1 is complete
    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheckErrors("cudaMemset failure");
    reduce_ws<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
    cudaCheckErrors("reduction warp shuffle kernel launch failure");
    // cuda processing sequence step 2 is complete
    // copy vector sums from device to host:
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("reduction warp shuffle kernel execution failure or cudaMemcpy H2D failure");
    //cuda processing sequence step 3 is complete
    if(*h_sum != (float)N)
    {
        printf("reduction warp shuffle sum incorrect!\n");
    }
    else
    {
        printf("reduction warp shuffle sum correct!\n");
    }
    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheckErrors("cudaMemset failure");

    // cuda processing sequence step 1 is complete
    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheckErrors("cudaMemset failure");
    reduce_ws_float4<<<(N / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
    cudaCheckErrors("reduction warp shuffle vec4 kernel launch failure");
    // cuda processing sequence step 2 is complete
    // copy vector sums from device to host:
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("reduction warp shuffle vec4 kernel execution failure or cudaMemcpy H2D failure");
    //cuda processing sequence step 3 is complete
    if(*h_sum != (float)N)
    {
        printf("reduction warp shuffle vec4 sum incorrect!\n");
    }
    else
    {
        printf("reduction warp shuffle vec4 sum correct!\n");
    }
    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheckErrors("cudaMemset failure");

    cudaFree(d_A);
    cudaFree(d_sum);

    delete [] h_A;
    delete h_sum;

    return 0;
}