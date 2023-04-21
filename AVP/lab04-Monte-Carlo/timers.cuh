#pragma once

#include <chrono>
#include <iostream>

#include "computation.cuh"

using namespace std;
using namespace chrono;

class TimerCPU {
public:
    ~TimerCPU() {
        const auto stop = high_resolution_clock::now();
        const auto elapsedTime = duration_cast<milliseconds>(stop - m_Start).count();
        cout << ", time=" << elapsedTime << "ms" << endl;
    }

private:
    time_point<high_resolution_clock> m_Start{high_resolution_clock::now()};
};

class TimerGPU {
public:
    TimerGPU() {
        CUDA_ASSERT(cudaEventCreate(&m_Start))
        CUDA_ASSERT(cudaEventRecord(m_Start))
    }

    ~TimerGPU() {
        cudaEvent_t stop;
        CUDA_ASSERT(cudaEventCreate(&stop))
        CUDA_ASSERT(cudaEventRecord(stop))
        CUDA_ASSERT(cudaEventSynchronize(stop))
        float elapsedTime;
        CUDA_ASSERT(cudaEventElapsedTime(&elapsedTime, m_Start, stop))
        cout << ", time=" << elapsedTime << "ms" << endl;
        CUDA_ASSERT(cudaEventDestroy(m_Start))
        CUDA_ASSERT(cudaEventDestroy(stop))
    }

private:
    cudaEvent_t m_Start{};
};
