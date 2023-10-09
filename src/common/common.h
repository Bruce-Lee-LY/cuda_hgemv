// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: common macro

#pragma once

#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "logging.h"
#include "util.h"

#define HGEMV_LIKELY(x) __builtin_expect(!!(x), 1)
#define HGEMV_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define HGEMV_CHECK(x)                    \
    do {                                  \
        if (HGEMV_UNLIKELY(!(x))) {       \
            HLOG("Check failed: %s", #x); \
            exit(EXIT_FAILURE);           \
        }                                 \
    } while (0)

#define HGEMV_CHECK_EQ(x, y) HGEMV_CHECK((x) == (y))
#define HGEMV_CHECK_NE(x, y) HGEMV_CHECK((x) != (y))
#define HGEMV_CHECK_LE(x, y) HGEMV_CHECK((x) <= (y))
#define HGEMV_CHECK_LT(x, y) HGEMV_CHECK((x) < (y))
#define HGEMV_CHECK_GE(x, y) HGEMV_CHECK((x) >= (y))
#define HGEMV_CHECK_GT(x, y) HGEMV_CHECK((x) > (y))

#define HGEMV_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName &) = delete;         \
    void operator=(const TypeName &) = delete

#define HGEMV_CHECK_CUDART_ERROR(_expr_)                                                          \
    do {                                                                                          \
        cudaError_t _ret_ = _expr_;                                                               \
        if (HGEMV_UNLIKELY(_ret_ != cudaSuccess)) {                                               \
            const char *_err_str_ = cudaGetErrorName(_ret_);                                      \
            int _rt_version_ = 0;                                                                 \
            cudaRuntimeGetVersion(&_rt_version_);                                                 \
            int _driver_version_ = 0;                                                             \
            cudaDriverGetVersion(&_driver_version_);                                              \
            HLOG("CUDA Runtime API error = %04d \"%s\", runtime version: %d, driver version: %d", \
                 static_cast<int>(_ret_), _err_str_, _rt_version_, _driver_version_);             \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

#define HGEMV_CHECK_CUBLAS_ERROR(_expr_)                                                                  \
    do {                                                                                                  \
        cublasStatus_t _ret_ = _expr_;                                                                    \
        if (HGEMV_UNLIKELY(_ret_ != CUBLAS_STATUS_SUCCESS)) {                                             \
            size_t _rt_version_ = cublasGetCudartVersion();                                               \
            HLOG("CUBLAS API error = %04d, runtime version: %zu", static_cast<int>(_ret_), _rt_version_); \
            exit(EXIT_FAILURE);                                                                           \
        }                                                                                                 \
    } while (0)
