#pragma once

#include <iostream>

using namespace std;

#define CUDA_ASSERT(cudaError)                          \
  if (cudaError != cudaSuccess) {                       \
    throw runtime_error{cudaGetErrorString(cudaError)}; \
  }

constexpr int PIXELS_PER_THREAD = 4;

[[nodiscard]]
__device__
inline void* rowPitched(void* rowPointer, size_t pitch, unsigned row) {
    return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(rowPointer) + row * pitch);
}
