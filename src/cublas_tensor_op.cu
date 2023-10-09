// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:49:00 on Mon, Oct 09, 2023
//
// Description: cublas tensor op hgemv

#include "common.h"

cublasHandle_t getCublasTensorOpHandle() {
    cublasHandle_t handle = nullptr;
    HGEMV_CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    HGEMV_CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    return handle;
}

void cublasTensorOp(half *A, half *B, half *C, size_t N, size_t K) {
    static cublasHandle_t handle = getCublasTensorOpHandle();
    static size_t M = 1;
    static float alpha = 1.0;
    static float beta = 0.0;

    HGEMV_CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A,
                                          CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
