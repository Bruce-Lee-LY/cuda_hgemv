// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:49:00 on Mon, Oct 09, 2023
//
// Description: thread naive hgemv

#include "common.h"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

__global__ void threadNaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
                                  size_t N, size_t K) {
    const size_t col = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (col >= N) {
        return;
    }

    float tmp = 0.0;
#pragma unroll
    for (size_t i = 0; i < K; ++i) {
        tmp += __half2float(A[i]) * __half2float(B[i + col * K]);
    }
    C[col] = __float2half(tmp);
}

void threadNaive(half *A, half *B, half *C, size_t N, size_t K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, THREADS_PER_BLOCK));

    threadNaiveKernel<<<grid, block>>>(A, B, C, N, K);
}
