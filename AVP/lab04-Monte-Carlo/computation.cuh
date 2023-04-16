#pragma once

#include <iostream>

#include "math.cuh"

using namespace std;

#define CUDA_ASSERT(cudaError)                                                 \
  if (cudaError != cudaSuccess) {                                              \
    cerr << cudaGetErrorString(cudaError) << ' ' << __FILE__ << ' '            \
         << __LINE__ << endl;                                                  \
    exit(1);                                                                   \
  }

constexpr int TOTAL_ITERATIONS = 100'000'000;
constexpr int ITERATIONS_PER_THREAD = 1000;

constexpr int TOTAL_USED_THREADS = TOTAL_ITERATIONS / ITERATIONS_PER_THREAD;

static_assert(TOTAL_ITERATIONS % ITERATIONS_PER_THREAD == 0);

[[nodiscard]]
float computeVolumeCPU();

[[nodiscard]]
float computeVolumeGPU();

[[nodiscard]]
constexpr float computeVolumeAccurate() {
    constexpr auto AB = computeVector(A, B);
    constexpr auto AC = computeVector(A, C);
    constexpr auto AD = computeVector(A, D);
    return computeDeterminant(AB, AC, AD) / 6.0f;
}

[[nodiscard]]
__device__
inline int warpReduceSum(int value);

[[nodiscard]]
__device__
inline int blockReduceSum(int value);

__host__
__device__
inline float adjustRandomNumber(float value, float min, float max);

__global__
void accumulateInnerPoints();
