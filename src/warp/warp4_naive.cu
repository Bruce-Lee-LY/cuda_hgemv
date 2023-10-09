// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:49:00 on Mon, Oct 09, 2023
//
// Description: warp4 naive hgemv

#include "common.h"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

#define COLS_PER_WARP 4
#define COLS_PER_BLOCK 16    // COLS_PER_WARP * WARPS_PER_BLOCK
#define THREADS_PER_GROUP 8  // WARP_SIZE / COLS_PER_WARP

__global__ void warp4NaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t N,
                                 size_t K) {
    const size_t group_id = threadIdx.x / THREADS_PER_GROUP;
    const size_t group_col = blockIdx.x * COLS_PER_BLOCK + group_id;
    if (group_col >= N) {
        return;
    }

    const size_t K_iters = div_ceil(K, THREADS_PER_GROUP);
    const size_t group_lane_id = threadIdx.x % THREADS_PER_GROUP;

    float tmp = 0.0;
#pragma unroll
    for (size_t i = 0; i < K_iters; ++i) {
        size_t A_idx = i * THREADS_PER_GROUP + group_lane_id;
        size_t B_idx = i * THREADS_PER_GROUP + group_lane_id + group_col * K;
        tmp += __half2float(A[A_idx]) * __half2float(B[B_idx]);
    }

    constexpr unsigned int mask = 0xffffffff;
#pragma unroll
    for (size_t i = THREADS_PER_GROUP / 2; i >= 1; i /= 2) {
        tmp += __shfl_xor_sync(mask, tmp, i);
    }

    if (group_lane_id == 0) {
        C[group_col] = __float2half(tmp);
    }
}

void warp4Naive(half *A, half *B, half *C, size_t N, size_t K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, COLS_PER_BLOCK));

    warp4NaiveKernel<<<grid, block>>>(A, B, C, N, K);
}
