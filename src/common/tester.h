// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:49:00 on Mon, Oct 09, 2023
//
// Description: tester

#pragma once

#include <memory>

#include "cuda_timer.h"
#include "matrix.h"

class Tester {
public:
    explicit Tester(size_t N = 256, size_t K = 128, size_t warmup_iterations = 1, size_t profiling_iterations = 10,
                    size_t sleep_duration = 100, bool enable_check = false)
        : m_N(N),
          m_K(K),
          m_warmup_iterations(warmup_iterations),
          m_profiling_iterations(profiling_iterations),
          m_sleep_duration(sleep_duration),
          m_enable_check(enable_check) {
        HGEMV_CHECK_GT(m_N, 0);
        HGEMV_CHECK_GT(m_K, 0);
        HGEMV_CHECK_GT(m_warmup_iterations, 0);
        HGEMV_CHECK_GT(m_profiling_iterations, 0);
        HGEMV_CHECK_GT(m_sleep_duration, 0);

        m_A = std::make_shared<Matrix>(1, m_K, "Vector A");
        HGEMV_CHECK(m_A);
        m_B = std::make_shared<Matrix>(m_K, m_N, "Matrix B");
        HGEMV_CHECK(m_B);
        m_C = std::make_shared<Matrix>(1, m_N, "Vector C");
        HGEMV_CHECK(m_C);
        m_base = std::make_shared<Matrix>(1, m_N, "Vector Base");
        HGEMV_CHECK(m_base);

        if (m_enable_check) {
            m_cuda_timer.start();
            cublas_tensor_op(m_A->getDevPtr(), m_B->getDevPtr(), m_base->getDevPtr(), m_N, m_K);
            HLOG("Cublas-Tensor-Op use: %.3f ms", m_cuda_timer.end());
            m_base->moveToHost();
            m_base->memSetDevice();
        }
    }

    ~Tester() {}

    template <typename Func>
    void evaluate(Func &&hgemv, const std::string &name) {
        HLOG("----------------- Evaluating %s -----------------", name.c_str());
        usleep(m_sleep_duration * 1000);
        m_C->tearUp(m_base.get());

        // warm up
        m_cuda_timer.start();
        for (size_t i = 0; i < m_warmup_iterations; ++i) {
            hgemv(m_A->getDevPtr(), m_B->getDevPtr(), m_C->getDevPtr(), m_N, m_K);
        }
        m_warmup_time = static_cast<double>(m_cuda_timer.end()) / static_cast<double>(m_warmup_iterations);
        HLOG("Warm up time: %.3f ms", m_warmup_time);

        if (m_enable_check) {
            m_C->moveToHost();
            m_C->checkValue(m_base.get());
        }

        profile(std::forward<Func>(hgemv), name);
    }

private:
    void cublas_tensor_op(half *A, half *B, half *C, size_t N, size_t K) {
        cublasHandle_t handle = nullptr;
        HGEMV_CHECK_CUBLAS_ERROR(cublasCreate(&handle));
        HGEMV_CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

        size_t M = 1;
        float alpha = 1.0;
        float beta = 0.0;

        HGEMV_CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A,
                                              CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        HGEMV_CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    }

    template <typename Func>
    void profile(Func &&hgemv, const std::string &name) {
        m_cuda_timer.start();
        for (size_t i = 0; i < m_profiling_iterations; ++i) {
            hgemv(m_A->getDevPtr(), m_B->getDevPtr(), m_C->getDevPtr(), m_N, m_K);
        }
        m_profiling_time = static_cast<double>(m_cuda_timer.end()) / static_cast<double>(m_profiling_iterations);
        m_throughput = static_cast<double>(m_N * m_K * 2) * 1e-12 / (static_cast<double>(m_profiling_time) * 1e-3);

        if ((std::abs(m_base_time) <= 1e-6) && (std::abs(m_base_throughput) <= 1e-6)) {
            m_base_time = m_profiling_time;
            m_base_throughput = m_throughput;
        }

        HLOG("%s exit, profiling time: %.4f ms (%.2f%%), throughput: %.6f TFLOPS (%.2f%%)", name.c_str(),
             m_profiling_time, m_profiling_time / m_base_time * 100, m_throughput,
             m_throughput / m_base_throughput * 100);
    }

    const size_t m_N = 256;
    const size_t m_K = 128;
    const size_t m_warmup_iterations = 1;
    const size_t m_profiling_iterations = 10;
    const size_t m_sleep_duration = 100;
    const bool m_enable_check = false;

    std::shared_ptr<Matrix> m_A = nullptr;     // row major, 1 * K
    std::shared_ptr<Matrix> m_B = nullptr;     // col major, K * N
    std::shared_ptr<Matrix> m_C = nullptr;     // row major, 1 * N
    std::shared_ptr<Matrix> m_base = nullptr;  // row major, 1 * N, base result, init vector C before each hgemv

    CudaTimer m_cuda_timer;

    double m_warmup_time = 0.0;
    double m_profiling_time = 0.0;
    double m_throughput = 0.0;
    double m_base_time = 0.0;        // cublas tensor op default
    double m_base_throughput = 0.0;  // cublas tensor op default

    HGEMV_DISALLOW_COPY_AND_ASSIGN(Tester);
};
