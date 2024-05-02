#include <stdio.h>
#include <math.h>

#define N 100000000
#define BLOCK_SIZE 256
#define GRID_SIZE 64
#define topk 20

__managed__ int source[N];
__managed__ int gpu_result[topk];
__managed__ int _1_pass_result[topk * GRID_SIZE];

// topK == 20
// source[N]:  1 + 2 + 3 + 4 + ...............N   
// cpu: for loop 
// gpu: 1 + 2 + 3 + 4 + ...............N    0 + 1 + 2 + 3 + 4[20] + 5 + 6 + 7 
// thread id step 0:  tid0:source[0][20] > source[4][20]? source[0] & source[4]-> source[0][20]
//                    tid1:source[1] + source[5] -> source[1]
//                    tid2:source[2] + source[6] -> source[2]
//                    tid4:source[4] + source[7] -> source[3]
//           step 1:  tid0: source[0] + source[2] -> source[0]
//                    tid1: source[1] + source[3] -> source[1]
//
//           step 2:  tid0: source[0] + source[1] -> source[0]
// thread id: blockDim.x * blockIdx.x + threadIdx.x + step * blockDim.x * GridDim.x
// thread 0: source[0, 8, 16, 24] sum -> shared memory

__device__ __host__ void insert_value(int *array, int k, int data)
{
    for(int i=0; i<k; i++)
    {
        if(array[i] == data)
        {
            return;
        }
    }
    if(data < array[k-1])
    {
        return;
    }
    //19, 18, 17, 16,.........4, 3, 2, 1, 0
    for(int i = k-2; i>=0; i--)
    {
        if(data > array[i])
        {
            array[i + 1] = array[i];
        }
        else
        {
            array[i+1] = data;
            return;
        }
    }
    
    array[0] = data;
}

__global__ void gpu_topk(int *input, int *output, int length, int k)
{
    __shared__ int ken[BLOCK_SIZE * topk];
    int top_array[topk];

    for(int i = 0; i<topk; i++)
    {
        top_array[i] = INT_MIN;
    }

    for(int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < length; idx += gridDim.x * blockDim.x)
    {
        insert_value(top_array, topk, input[idx]);
    }
    for(int i =0; i<topk; i++)
    {
        ken[topk * threadIdx.x + i] = top_array[i];
    }
    __syncthreads();

    for(int i = BLOCK_SIZE/2; i>=1; i/=2)
    {
        if(threadIdx.x < i)
        {
            for(int m=0; m<topk; m++)
            {
                insert_value(top_array, topk, ken[topk *(threadIdx.x + i) + m]);
            }
        }
        __syncthreads();
        if(threadIdx.x < i)
        {
            for(int m=0; m<topk; m++)
            {
                ken[topk * threadIdx.x + m] = top_array[m];
            }
        }
        __syncthreads();
    }
    if(blockIdx.x * blockDim.x < length)
    {
        if(threadIdx.x == 0 )
        {
            for(int i =0; i < topk; i++)
            {
                output[topk * blockIdx.x + i] = ken[i];
            }
        }
    }


}

void cpu_topk(int *input, int *output, int length, int k)
{
    for(int i =0; i< length; i++)
    {
        insert_value(output, k, input[i]);
    }
}

int main()
{
    printf("Init source data...........\n");
    for(int i=0; i<N; i++)
    {
        source[i] = rand();
    }

    printf("Complete init source data.....\n");
    cudaEvent_t start, stop_gpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_gpu);
    cudaEventCreate(&stop_cpu);

    cudaEventRecord(start);
    cudaEventSynchronize(start);
    printf("GPU Run **************\n");
    for(int i =0; i<20; i++)
    {
        gpu_topk<<<GRID_SIZE, BLOCK_SIZE>>>(source, _1_pass_result, N, topk);

        gpu_topk<<<1, BLOCK_SIZE>>>(_1_pass_result, gpu_result, topk * GRID_SIZE, topk);

        cudaDeviceSynchronize();
    }
    printf("GPU Complete!!!\n");
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    int cpu_result[topk] ={0};
    printf("CPU RUN***************\n");
    cpu_topk(source, cpu_result, N, topk);
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    printf("CPU Complete!!!!!\n");

    float time_cpu, time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop_gpu);
    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);

    bool error = false;
    for(int i =0; i<topk; i++)
    {
        printf("CPU top%d: %d; GPU top%d: %d;\n", i+1, cpu_result[i], i+1, gpu_result[i]);
        if(fabs(gpu_result[i] - cpu_result[i]) > 0)
        {
            error = true;
        }
    }
    printf("Result: %s\n", (error?"Error":"Pass"));
    printf("CPU time: %.2f ms; GPU time: %.2f ms\n", time_cpu, (time_gpu/20.0));
}