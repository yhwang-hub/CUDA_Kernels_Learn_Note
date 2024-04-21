#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(value)  *(float4*)(&(value))
#define INT4(value)    *(int4*)(&(value))
#define WARP_SIZE 32

#define checkCudaErrors(func)               \
{                                   \
    cudaError_t e = (func);         \
    if(e != cudaSuccess)                                        \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));        \
}

__global__ void sgemm_register_tile_vec4(
                    float* __restrict__ global_a, float* __restrict__ global_b, float* __restrict__ global_c,
                    int M, int N, int K)
{
    constexpr int block_m  = 128;
    constexpr int block_n  = 128;
    constexpr int block_k  = 8;
    constexpr int thread_m = 8;
    constexpr int thread_n = 8;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = ty * blockDim.x + tx;

    __shared__ float shared_a[2][block_k][block_m];
    __shared__ float shared_b[2][block_k][block_n];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[thread_m];
    float r_comp_b[thread_n];
    float r_c[thread_m][thread_n] = {0.0f};

    int sm_am = tid / 2;
    int sm_ak = (tid % 2 == 0) ? 0 : 4;
    int sm_bk = tid / 32;
    int sm_bn = (tid % 32) * 4;

    int gm_am = by * block_m + sm_am;
    int gm_bn = bx * block_n + sm_bn;

    {
        int gm_ak = sm_ak;
        int gm_a_offset = gm_am * K + gm_ak;
        int gm_bk = sm_bk;
        int gm_b_offset = gm_bk * N + gm_bn;
        FLOAT4(r_load_a[0]) = FLOAT4(global_a[gm_a_offset]);
        FLOAT4(r_load_b[0]) = FLOAT4(global_b[gm_b_offset]);

        shared_a[0][sm_ak    ][sm_am] = r_load_a[0];
        shared_a[0][sm_ak + 1][sm_am] = r_load_a[1];
        shared_a[0][sm_ak + 2][sm_am] = r_load_a[2];
        shared_a[0][sm_ak + 3][sm_am] = r_load_a[3];
        FLOAT4(shared_b[0][sm_bk][sm_bn]) = FLOAT4(r_load_b[0]);
    }

    for (int bk = 1; bk < (K + block_k - 1) / block_k; bk++)
    {
        __syncthreads();
        int sm_sel = (bk - 1) & 1;
        int sm_sel_next = bk & 1;

        int gm_ak = bk * block_k + sm_ak;
        int gm_a_offset = gm_am * K + gm_ak;
        int gm_bk = bk * block_k + sm_bk;
        int gm_b_offset = gm_bk * N + gm_bn;
        FLOAT4(r_load_a[0]) = FLOAT4(global_a[gm_a_offset]);
        FLOAT4(r_load_b[0]) = FLOAT4(global_b[gm_b_offset]);

        #pragma unroll
        for (int k = 0; k < block_k; k++)
        {
            FLOAT4(r_comp_a[0]) = FLOAT4(shared_a[sm_sel][k][ty * thread_m / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(shared_a[sm_sel][k][ty * thread_m / 2 + block_m / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(shared_b[sm_sel][k][tx * thread_n / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(shared_b[sm_sel][k][tx * thread_n / 2 + block_n / 2]);

            #pragma unroll
            for (int m = 0; m < thread_m; m++)
            {
                #pragma unroll
                for (int n = 0; n < thread_n; n++)
                {
                    r_c[m][n] += r_comp_a[m] * r_comp_b[n];
                }
            }
        }

        shared_a[sm_sel_next][sm_ak    ][sm_am] = r_load_a[0];
        shared_a[sm_sel_next][sm_ak + 1][sm_am] = r_load_a[1];
        shared_a[sm_sel_next][sm_ak + 2][sm_am] = r_load_a[2];
        shared_a[sm_sel_next][sm_ak + 3][sm_am] = r_load_a[3];
        FLOAT4(shared_b[sm_sel_next][sm_bk][sm_bn]) = FLOAT4(r_load_b[0]);
    }
    // __syncthreads();

    #pragma unroll
    for (int k = 0; k < block_k; k++)
    {
        FLOAT4(r_comp_a[0]) = FLOAT4(shared_a[1][k][ty * thread_m / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(shared_a[1][k][ty * thread_m / 2 + block_m / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(shared_b[1][k][tx * thread_n / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(shared_b[1][k][tx * thread_n / 2 + block_n / 2]);

        #pragma unroll
        for (int m = 0; m < thread_m; m++)
        {
            #pragma unroll
            for (int n = 0; n < thread_n; n++)
            {
                r_c[m][n] += r_comp_a[m] * r_comp_b[n];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < thread_m / 2; i++)
    {
        int gm_cm = by * block_m + ty * thread_m / 2 + i;
        int gm_cn = bx * block_n + tx * thread_n / 2;
        int gm_c_offset = gm_cm * N + gm_cn;
        FLOAT4(global_c[gm_c_offset]) = FLOAT4(r_c[i][0]);
        FLOAT4(global_c[gm_c_offset + block_n / 2]) = FLOAT4(r_c[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < thread_m / 2; i++)
    {
        int gm_cm = by * block_m + block_m / 2 + ty * thread_m / 2 + i;
        int gm_cn = bx * block_n               + tx * thread_n / 2;
        int gm_c_offset = gm_cm * N + gm_cn;
        FLOAT4(global_c[gm_c_offset]) = FLOAT4(r_c[i + thread_m / 2][0]);
        FLOAT4(global_c[gm_c_offset + block_n / 2]) = FLOAT4(r_c[i + thread_m / 2][4]);
    }
}

template <
    const int BLOCK_SIZE_M,  // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X, // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
>
__global__ void sgemm_v3(
    float* __restrict__ A, float* __restrict__ B, float* __restrict__ C,
    const int M, const int N, const int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X];
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_Y; i++)
    {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_X; j++)
        {
            accum[i][j] = 0.0f;
        }
    }
    
    // registers for A and B
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];

    // registers load global memory
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    A = &A[(BLOCK_SIZE_M * by) * K];
    B = &B[ BLOCK_SIZE_N * bx ];

    //load index of the tile
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    // warp_id * 8 + (lane_id / 16)*4; // (warp_id/4)*32 + ((lane_id%16)/2)*4;
    const int a_tile_index = warp_id / 2 * 16 + lane_id / 8 * 4;
    // (lane_id % 16) * 4; // (warp_id%4)*16 + (lane_id/16)*8 + (lane_id%2)*4;
    const int b_tile_index = warp_id % 2 * 32 + lane_id % 8 * 4;

    //transfer first tile from global mem to shared mem
    // load A from global memory to shared memory
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
    {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index])= FETCH_FLOAT4(
            A[OFFSET(A_TILE_ROW_START + i, A_TILE_COL, K)]
        );
        As[0][A_TILE_COL    ][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index    ];
        As[0][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
        As[0][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
        As[0][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];
    }
    
    // load B from global memory to shared memory
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
    {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(
            B[OFFSET(B_TILE_ROW_START + i, B_TILE_COL, N)]
        );
    }
    __syncthreads();

    // load A from shared memory to register
    FETCH_FLOAT4(frag_a[0][0]) = FETCH_FLOAT4(As[0][0][a_tile_index     ]);
    FETCH_FLOAT4(frag_a[0][4]) = FETCH_FLOAT4(As[0][0][a_tile_index + 64]);

    // load B from shared memory to register
    FETCH_FLOAT4(frag_b[0][0]) = FETCH_FLOAT4(Bs[0][0][b_tile_index     ]);
    FETCH_FLOAT4(frag_b[0][4]) = FETCH_FLOAT4(Bs[0][0][b_tile_index + 64]);

    int write_stage_idx = 1;
    int tile_idx = 0;
    do
    {
        // next tile index
        tile_idx += BLOCK_SIZE_K;
        // load next tile from global mem
        if (tile_idx < K)
        {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
            {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(
                    A[OFFSET(A_TILE_ROW_START + i, A_TILE_COL + tile_idx, K)]
                );
            }

            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
            {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(
                    B[OFFSET(B_TILE_ROW_START + tile_idx + i, B_TILE_COL, N)]
                );
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K - 1; j++)
        {
            // load next tile from shared mem to register 
            // load A from shared memory to register
            FETCH_FLOAT4(frag_a[(j + 1) % 2][0]) = FETCH_FLOAT4(As[load_stage_idx][j + 1][a_tile_index     ]);
            FETCH_FLOAT4(frag_a[(j + 1) % 2][4]) = FETCH_FLOAT4(As[load_stage_idx][j + 1][a_tile_index + 64]);
            // load B from shared memory to register
            FETCH_FLOAT4(frag_b[(j + 1) % 2][0]) = FETCH_FLOAT4(Bs[load_stage_idx][j + 1][b_tile_index     ]);
            FETCH_FLOAT4(frag_b[(j + 1) % 2][4]) = FETCH_FLOAT4(Bs[load_stage_idx][j + 1][b_tile_index + 64]);

            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++)
            {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++)
                {
                    accum[thread_y][thread_x] += frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
                }
            }
        }

        if (tile_idx < K)
        {
            // load A from global memory to shared memory
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
            {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL    ][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index    ];
                As[write_stage_idx][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
                As[write_stage_idx][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
                As[write_stage_idx][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];
            }

            // load B from global memory to shared memory
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
            {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }

            // use double buffer, only need one sync
            __syncthreads();
            // switch
            write_stage_idx ^= 1;
        }

        // load first tile from shared mem to register of next iter
        // load A from shared memory to register
        FETCH_FLOAT4(frag_a[0][0]) = FETCH_FLOAT4(As[load_stage_idx ^ 1][0][a_tile_index]);
        FETCH_FLOAT4(frag_a[0][4]) = FETCH_FLOAT4(As[load_stage_idx ^ 1][0][a_tile_index + 64]);
        // load B from shared memory to register
        FETCH_FLOAT4(frag_b[0][0]) = FETCH_FLOAT4(Bs[load_stage_idx ^ 1][0][b_tile_index]);
        FETCH_FLOAT4(frag_b[0][4]) = FETCH_FLOAT4(Bs[load_stage_idx ^ 1][0][b_tile_index + 64]);

        // compute C THREAD_SIZE_X x THREAD_SIZE_Y
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++)
        {
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++)
            {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    } while (tile_idx < K);
    
    const int c_block_row = a_tile_index;
    const int c_block_col = b_tile_index;

    //store C00 block
    for (int i = 0; i < 4; i++)
    {
        FETCH_FLOAT4(C[OFFSET(
            BLOCK_SIZE_M * by + c_block_row + i,
            BLOCK_SIZE_N * bx + c_block_col,
            N)]) = FETCH_FLOAT4(accum[i][0]);
    }

    //store C01 block
    for (int i = 0; i < 4; i++)
    {
        FETCH_FLOAT4(C[OFFSET(
            BLOCK_SIZE_M * by + c_block_row + i,
            BLOCK_SIZE_N * bx + c_block_col + 64,
            N)]) = FETCH_FLOAT4(accum[i][4]);
    }

    //store C10 block
    for (int i = 0; i < 4; i++)
    {
        FETCH_FLOAT4(C[OFFSET(
            BLOCK_SIZE_M * by + c_block_row + 64 + i,
            BLOCK_SIZE_N * bx + c_block_col,
            N)]) = FETCH_FLOAT4(accum[i + 4][0]);
    }

    //store C11 block
    for (int i = 0; i < 4; i++)
    {
        FETCH_FLOAT4(C[OFFSET(
            BLOCK_SIZE_M * by + c_block_row + 64 + i,
            BLOCK_SIZE_N * bx + c_block_col + 64,
            N)]) = FETCH_FLOAT4(accum[i + 4][4]);
    }
}

int main(int argc, char** argv)
{
    size_t M = 4096;
    size_t K = 4096;
    size_t N = 4096;

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    
    double duration[2] = {0, 0};
    double GFLOPS[2] = {0, 0};        //（全部大写）是floating-point operations per second的缩写，意指每秒浮点运算次数。用来衡量硬件的性能
    double GFLOPs = 2.0 * M * N * K;  //(s小写)是floating point of operations的缩写，是浮点运算次数，可以用来衡量算子/模型计算量

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 128;
    const int BLOCK_SIZE_K = 8;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    for( int i = 0; i < M * K; i++ )
    {
        h_A[i] = i / 666;
    }

    for( int i = 0; i < K * N; i++ )
    {
        h_B[i] = i % 666;
    }

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float durationTotal = 0;
    int iteration = 10;
    
    // warm-up
    for (int run = 0 ; run < iteration; run ++ )
    {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
        sgemm_register_tile_vec4<<<dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
        // sgemm_v3<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
        //     <<<dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
    }

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < iteration; run ++ )
    {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
        sgemm_register_tile_vec4<<<dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
        // sgemm_v3<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
        //     <<<dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&durationTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    duration[0] = durationTotal / iteration;
    GFLOPS[0] = (GFLOPs * 1.0e-9f) / (duration[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        GFLOPS[0],
        duration[0],
        GFLOPs);

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < iteration; run ++ )
    {
        cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, 
            d_A, K, d_B, N, &beta, d_C, N
        );
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&durationTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    duration[1] = durationTotal / iteration;
    GFLOPS[1] = (GFLOPs * 1.0e-9f) / (duration[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        GFLOPS[1],
        duration[1],
        GFLOPs);

    cublasDestroy(blas_handle); 
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++)
    {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(h_C[i] - h_C1[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_C[i], h_C1[col * M + row], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("My Gemm to CuBlas implementation ratio = %f\n", GFLOPS[0] / GFLOPS[1]);
    
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);

    return 0;
}