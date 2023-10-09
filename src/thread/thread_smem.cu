// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:49:00 on Mon, Oct 09, 2023
//
// Description: thread smem hgemv

#include "common.h"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

__global__ void threadSmemKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t N,
                                 size_t K) {
    extern __shared__ half A_smem[];
    size_t A_smem_iters = div_ceil(K, THREADS_PER_BLOCK);
#pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        size_t idx = i * THREADS_PER_BLOCK + threadIdx.x;
        A_smem[idx] = A[idx];
    }

    __syncthreads();

    const size_t col = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (col >= N) {
        return;
    }

    float tmp = 0.0;
#pragma unroll
    for (size_t i = 0; i < K; ++i) {
        tmp += __half2float(A_smem[i]) * __half2float(B[i + col * K]);
    }
    C[col] = __float2half(tmp);
}

size_t initThreadSmem(size_t K) {
    int dev_id = 0;
    HGEMV_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMV_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size = K * sizeof(half);
    HLOG("smem_max_size: %.0f KBytes (%zu bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMV_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMV_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(threadSmemKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void threadSmem(half *A, half *B, half *C, size_t N, size_t K) {
    static size_t smem_max_size = initThreadSmem(K);

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, THREADS_PER_BLOCK));

    threadSmemKernel<<<grid, block, smem_max_size>>>(A, B, C, N, K);
}
