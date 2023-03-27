#include "computation.hpp"

#include <random>

#include <curand_kernel.h>

__device__
constexpr Plane ABC{A, B, C};
__device__
constexpr Plane ABD{A, B, D};
__device__
constexpr Plane BCD{B, C, D};
__device__
constexpr Plane CAD{C, A, D};

float computeVolumeCPU() {
    random_device rd;
    mt19937 generator{rd()};
    uniform_real_distribution<float> distribution{0.0f, 1.0f};

    int innerPoints = 0;
    for (int i = 0; i < TOTAL_ITERATIONS; ++i) {
        const Point point{
                adjustRandomNumber(distribution(generator), MIN_X, MAX_X),
                adjustRandomNumber(distribution(generator), MIN_Y, MAX_Y),
                adjustRandomNumber(distribution(generator), MIN_Z, MAX_Z)
        };
        if (ABC.isOnGoodSide(point) and ABD.isOnGoodSide(point) and BCD.isOnGoodSide(point) and
            CAD.isOnGoodSide(point)) {
            ++innerPoints;
        }
    }
    return CUBOID_VOLUME * static_cast<float>(innerPoints) / static_cast<float>(TOTAL_ITERATIONS);
}

__managed__
unsigned long long innerPointsAccumulator = 0;

float computeVolumeGPU() {
    constexpr int THREADS_PER_BLOCK = 128;
    constexpr int BLOCKS_PER_GRID = (TOTAL_USED_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    accumulateInnerPoints<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
    CUDA_ASSERT(cudaDeviceSynchronize())
    return CUBOID_VOLUME * static_cast<float>(innerPointsAccumulator) / static_cast<float>(TOTAL_ITERATIONS);
}

constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int WARP_SIZE = 32;
constexpr int MAX_WARPS_PER_BLOCK = MAX_THREADS_PER_BLOCK / WARP_SIZE;

__device__
int warpReduceSum(int value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__
int blockReduceSum(int value) {
    __shared__ int reduceBuffer[MAX_WARPS_PER_BLOCK];

    const unsigned warpLine = threadIdx.x % WARP_SIZE;
    const unsigned warpId = threadIdx.x / WARP_SIZE;

    value = warpReduceSum(value);
    if (warpLine == 0) {
        reduceBuffer[warpId] = value;
    }
    __syncthreads();

    // Loads values back only in the first warp of the first block
    value = (threadIdx.x < blockDim.x / WARP_SIZE) ? reduceBuffer[warpLine] : 0;
    if (warpId == 0) {
        value = warpReduceSum(value);
    }
    return value;
}

__host__
__device__
float adjustRandomNumber(float value, float min, float max) {
    return value * (max - min) + min;
}

__global__
void accumulateInnerPoints() {
    const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= TOTAL_USED_THREADS) {
        return;
    }

    constexpr int SEED = 1337;
    curandState randomState;
    curand_init(SEED, threadId, 0, &randomState);

    int innerPoints = 0;
    for (int i = 0; i < ITERATIONS_PER_THREAD; ++i) {
        const Point point{
                adjustRandomNumber(curand_uniform(&randomState), MIN_X, MAX_X),
                adjustRandomNumber(curand_uniform(&randomState), MIN_Y, MAX_Y),
                adjustRandomNumber(curand_uniform(&randomState), MIN_Z, MAX_Z),
        };
        if (ABC.isOnGoodSide(point) and ABD.isOnGoodSide(point) and BCD.isOnGoodSide(point)
            and CAD.isOnGoodSide(point)) {
            ++innerPoints;
        }
    }

    const int blockInnerPoints = blockReduceSum(innerPoints);
    if (threadIdx.x == 0) {
        atomicAdd(&innerPointsAccumulator, blockInnerPoints);
    }
}
