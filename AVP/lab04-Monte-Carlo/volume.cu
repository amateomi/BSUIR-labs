#include <iostream>

#include <curand_kernel.h>

#define CUDA_ASSERT(cudaError)                                                               \
    if (cudaError != cudaSuccess) {                                                          \
        cerr << cudaGetErrorString(cudaError) << ' ' << __FILE__ << ' ' << __LINE__ << endl; \
        exit(1);                                                                             \
    }

using namespace std;

constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int WARP_SIZE = 32;
constexpr int MAX_WARPS_PER_BLOCK = MAX_THREADS_PER_BLOCK / WARP_SIZE;

[[nodiscard]]
__device__
inline int warpReduceSum(int value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template<int REDUCE_BUFFER_SIZE = MAX_WARPS_PER_BLOCK>
[[nodiscard]]
__device__
inline int blockReduceSum(int value) {
    __shared__ int reduceBuffer[REDUCE_BUFFER_SIZE];

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

constexpr int OUTER_SQUARE_AREA = 22 * 22;
constexpr int RADIUS = 11;

constexpr int TOTAL_ITERATIONS = 10'000'000;
constexpr int ITERATIONS_PER_THREAD = 100;
static_assert(TOTAL_ITERATIONS % ITERATIONS_PER_THREAD == 0);

constexpr int TOTAL_USED_THREADS = TOTAL_ITERATIONS / ITERATIONS_PER_THREAD;

__managed__
unsigned long long innerPointsAccumulator = 0;

// Maps [0.0, 1.0] to [-RADIUS, RADIUS]
template<int FACTOR = RADIUS>
[[nodiscard]]
__device__
inline float normalize(float value) {
    return (value - 0.5f) * 2 * FACTOR;
}

template<int RANDOM_SEED = 1337>
__global__
void calculateArea() {
    const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= TOTAL_USED_THREADS) {
        return;
    }

    curandState randomState;
    curand_init(RANDOM_SEED, threadId, 0, &randomState);

    int innerPoints = 0;
    for (int i = 0; i < ITERATIONS_PER_THREAD; ++i) {
        const float2 point{
                normalize(curand_uniform(&randomState)),
                normalize(curand_uniform(&randomState))
        };
        if (point.x * point.x + point.y * point.y <= RADIUS * RADIUS) {
            ++innerPoints;
        }
    }

    const int blockInnerPoints = blockReduceSum(innerPoints);
    if (threadIdx.x == 0) {
        atomicAdd(&innerPointsAccumulator, blockInnerPoints);
    }
}

int main() {
    constexpr int THREADS_PER_BLOCK = 128;
    constexpr int BLOCKS_PER_GRID = (TOTAL_USED_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    calculateArea<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
    CUDA_ASSERT(cudaDeviceSynchronize())
    cout << OUTER_SQUARE_AREA * innerPointsAccumulator / TOTAL_ITERATIONS;
}
