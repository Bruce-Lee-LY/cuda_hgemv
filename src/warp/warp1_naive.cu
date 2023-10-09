// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:49:00 on Mon, Oct 09, 2023
//
// Description: warp1 naive hgemv

#include "common.h"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

__global__ void warp1NaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t N,
                                 size_t K) {
    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t warp_col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (warp_col >= N) {
        return;
    }

    const size_t K_iters = div_ceil(K, WARP_SIZE);
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    float tmp = 0.0;
#pragma unroll
    for (size_t i = 0; i < K_iters; ++i) {
        size_t A_idx = i * WARP_SIZE + lane_id;
        size_t B_idx = i * WARP_SIZE + lane_id + warp_col * K;
        tmp += __half2float(A[A_idx]) * __half2float(B[B_idx]);
    }

    constexpr unsigned int mask = 0xffffffff;
#pragma unroll
    for (size_t i = WARP_SIZE / 2; i >= 1; i /= 2) {
        tmp += __shfl_xor_sync(mask, tmp, i);
    }

    if (lane_id == 0) {
        C[warp_col] = __float2half(tmp);
    }
}

void warp1Naive(half *A, half *B, half *C, size_t N, size_t K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, WARPS_PER_BLOCK));

    warp1NaiveKernel<<<grid, block>>>(A, B, C, N, K);
}
