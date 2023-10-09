// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:49:00 on Mon, Oct 09, 2023
//
// Description: hgemv main

#include "gflags/gflags.h"
#include "omp.h"
#include "tester.h"

#define HGEMV_FUNC(name) void name(half *A, half *B, half *C, size_t N, size_t K)

HGEMV_FUNC(cublasTensorOp);

HGEMV_FUNC(threadNaive);
HGEMV_FUNC(threadSmem);

HGEMV_FUNC(warp1Naive);
HGEMV_FUNC(warp1Smem);
HGEMV_FUNC(warp2Naive);
HGEMV_FUNC(warp2Smem);
HGEMV_FUNC(warp4Naive);
HGEMV_FUNC(warp4Smem);
HGEMV_FUNC(warp8Naive);
HGEMV_FUNC(warp8Smem);
HGEMV_FUNC(warp16Naive);
HGEMV_FUNC(warp16Smem);

DEFINE_uint32(N, 256, "N");
DEFINE_uint32(K, 128, "K");
DEFINE_uint32(warmup_iterations, 1, "warmup iteration numbers and average the result");
DEFINE_uint32(profiling_iterations, 10, "profiling iteration numbers and average the result");
DEFINE_uint32(sleep_duration, 100, "sleep_milliseconds between profiling");
DEFINE_bool(enable_check, false, "check the GPU result against the cublas result");
DEFINE_uint32(cpu_procs, omp_get_num_procs(), "processor num used of CPU");
DEFINE_uint32(gpu_rank, 0, "the used GPU rank");

int main(int argc, char *argv[]) {
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    omp_set_num_threads(FLAGS_cpu_procs);
    HGEMV_CHECK_CUDART_ERROR(cudaSetDevice(FLAGS_gpu_rank));

    cudaDeviceProp dev_prop;
    HGEMV_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, FLAGS_gpu_rank));
    HLOG("CUDA HGEMV start with %u CPU processes on the %u-th GPU: %s", FLAGS_cpu_procs, FLAGS_gpu_rank, dev_prop.name);

    int driver_version = 0;
    int runtime_version = 0;
    HGEMV_CHECK_CUDART_ERROR(cudaDriverGetVersion(&driver_version));
    HGEMV_CHECK_CUDART_ERROR(cudaRuntimeGetVersion(&runtime_version));
    HLOG("CUDA driver version / runtime version: %d.%d / %d.%d", driver_version / 1000, (driver_version % 100) / 10,
         runtime_version / 1000, (runtime_version % 100) / 10);
    HLOG("CUDA capability major/minor version number: %d.%d", dev_prop.major, dev_prop.minor);
    HLOG("%d multiprocessors, %d CUDA cores/MP: %d CUDA cores", dev_prop.multiProcessorCount,
         convert_SM_to_cores(dev_prop.major, dev_prop.minor),
         convert_SM_to_cores(dev_prop.major, dev_prop.minor) * dev_prop.multiProcessorCount);
    HLOG("GPU max clock rate: %.0f MHz (%0.2f GHz)", static_cast<double>(dev_prop.clockRate) * 1e-3,
         static_cast<double>(dev_prop.clockRate) * 1e-6);
    HLOG("Memory clock rate: %.0f MHz (%0.2f GHz)", static_cast<double>(dev_prop.memoryClockRate) * 1e-3,
         static_cast<double>(dev_prop.memoryClockRate) * 1e-6);
    HLOG("Memory bus width: %d-bit", dev_prop.memoryBusWidth);
    HLOG("Total amount of global memory: %.0f MBytes (%zu Bytes)",
         static_cast<double>(dev_prop.totalGlobalMem) / 1048576, dev_prop.totalGlobalMem);
    HLOG("Total amount of constant memory: %.0f KBytes (%zu Bytes)", static_cast<double>(dev_prop.totalConstMem) / 1024,
         dev_prop.totalConstMem);
    HLOG("Total amount of shared memory per block: %.0f KBytes (%zu Bytes)",
         static_cast<double>(dev_prop.sharedMemPerBlock) / 1024, dev_prop.sharedMemPerBlock);
    HLOG("Total shared memory per multiprocessor: %.0f KBytes (%zu Bytes)",
         static_cast<double>(dev_prop.sharedMemPerMultiprocessor) / 1024, dev_prop.sharedMemPerMultiprocessor);
    HLOG("L2 cache size: %.0f KBytes (%d Bytes)", static_cast<double>(dev_prop.l2CacheSize) / 1024,
         dev_prop.l2CacheSize);
    HLOG("Total number of registers available per block: %d", dev_prop.regsPerBlock);
    HLOG("Warp size: %d", dev_prop.warpSize);
    HLOG("Max number of threads per multiprocessor: %d", dev_prop.maxThreadsPerMultiProcessor);
    HLOG("Max number of threads per block: %d", dev_prop.maxThreadsPerBlock);
    HLOG("Max dimension size of a thread block (x,y,z): (%d, %d, %d)", dev_prop.maxThreadsDim[0],
         dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
    HLOG("Max dimension size of a grid size (x,y,z): (%d, %d, %d)", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1],
         dev_prop.maxGridSize[2]);

    HLOG("A (1 x %u) * B (%u x %u) = C (1 x %u)", FLAGS_K, FLAGS_K, FLAGS_N, FLAGS_N);
    HLOG("Profiling: warmup iterations: %u, profiling iterations: %u, sleep duration: %u ms, enable check: %d",
         FLAGS_warmup_iterations, FLAGS_profiling_iterations, FLAGS_sleep_duration, FLAGS_enable_check);

    Tester tester(FLAGS_N, FLAGS_K, FLAGS_warmup_iterations, FLAGS_profiling_iterations, FLAGS_sleep_duration,
                  FLAGS_enable_check);
    tester.evaluate(cublasTensorOp, "Cublas-Tensor-Op");

    tester.evaluate(threadNaive, "Thread-Naive");
    tester.evaluate(threadSmem, "Thread-Smem");

    tester.evaluate(warp1Naive, "Warp1-Naive");
    tester.evaluate(warp1Smem, "Warp1-Smem");
    tester.evaluate(warp2Naive, "Warp2-Naive");
    tester.evaluate(warp2Smem, "Warp2-Smem");
    tester.evaluate(warp4Naive, "Warp4-Naive");
    tester.evaluate(warp4Smem, "Warp4-Smem");
    tester.evaluate(warp8Naive, "Warp8-Naive");
    tester.evaluate(warp8Smem, "Warp8-Smem");
    tester.evaluate(warp16Naive, "Warp16-Naive");
    tester.evaluate(warp16Smem, "Warp16-Smem");

    GFLAGS_NAMESPACE::ShutDownCommandLineFlags();

    HLOG("Done");

    return 0;
}
