// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:49:00 on Mon, Oct 09, 2023
//
// Description: warp1 smem hgemv

#include "common.h"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

__global__ void warp1SmemKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t N,
                                size_t K) {
    extern __shared__ half A_smem[];
    size_t A_smem_iters = div_ceil(K, THREADS_PER_BLOCK);
#pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        size_t idx = i * THREADS_PER_BLOCK + threadIdx.x;
        A_smem[idx] = A[idx];
    }

    __syncthreads();

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
        tmp += __half2float(A_smem[A_idx]) * __half2float(B[B_idx]);
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

size_t initWarp1Smem(size_t K) {
    int dev_id = 0;
    HGEMV_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMV_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size = K * sizeof(half);
    HLOG("smem_max_size: %.0f KBytes (%zu bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMV_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMV_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(warp1SmemKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void warp1Smem(half *A, half *B, half *C, size_t N, size_t K) {
    static size_t smem_max_size = initWarp1Smem(K);

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, WARPS_PER_BLOCK));

    warp1SmemKernel<<<grid, block, smem_max_size>>>(A, B, C, N, K);
}
