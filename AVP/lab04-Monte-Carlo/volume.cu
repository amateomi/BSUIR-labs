#include <algorithm>
#include <iostream>
#include <random>

#include <curand_kernel.h>

#define CUDA_ASSERT(cudaError)                                                                                         \
    if (cudaError != cudaSuccess) {                                                                                    \
        cerr << cudaGetErrorString(cudaError) << ' ' << __FILE__ << ' ' << __LINE__ << endl;                           \
        exit(1);                                                                                                       \
    }

using namespace std;

constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int WARP_SIZE = 32;
constexpr int MAX_WARPS_PER_BLOCK = MAX_THREADS_PER_BLOCK / WARP_SIZE;

[[nodiscard]] __device__ inline int warpReduceSum(int value)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int REDUCE_BUFFER_SIZE = MAX_WARPS_PER_BLOCK> [[nodiscard]] __device__ inline int blockReduceSum(int value)
{
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

constexpr int TOTAL_ITERATIONS = 100'000'000;
constexpr int ITERATIONS_PER_THREAD = 100;
static_assert(TOTAL_ITERATIONS % ITERATIONS_PER_THREAD == 0);

constexpr int TOTAL_USED_THREADS = TOTAL_ITERATIONS / ITERATIONS_PER_THREAD;

using Point = float3;

constexpr Point A { 0, 0, -1 };
constexpr Point B { -1, 0, 0 };
constexpr Point C { 0, -1, 0 };
constexpr Point D { 1, 1, 1 };

constexpr float MIN_X = min({ A.x, B.x, C.x, D.x });
constexpr float MAX_X = max({ A.x, B.x, C.x, D.x });

constexpr float MIN_Y = min({ A.y, B.y, C.y, D.y });
constexpr float MAX_Y = max({ A.y, B.y, C.y, D.y });

constexpr float MIN_Z = min({ A.z, B.z, C.z, D.z });
constexpr float MAX_Z = max({ A.z, B.z, C.z, D.z });

[[nodiscard]] constexpr float computeDeterminant(float2 row1, float2 row2) { return row1.x * row2.y - row1.y * row2.x; }

struct Plane {
public:
    constexpr Plane(Point p1, Point p2, Point p3)
        : DETERMINANTS { computeDeterminant({ p2.y - p1.y, p2.z - p1.z }, { p3.y - p1.y, p3.z - p1.z }),
            computeDeterminant({ p2.x - p1.x, p2.z - p1.z }, { p3.x - p1.x, p3.z - p1.z }),
            computeDeterminant({ p2.x - p1.x, p2.y - p1.y }, { p3.x - p1.x, p3.y - p1.y }) }
        , a { DETERMINANTS.x }
        , b { -DETERMINANTS.y }
        , c { DETERMINANTS.z }
        , d { -p1.x * DETERMINANTS.x + p1.y * DETERMINANTS.y - p1.z * DETERMINANTS.z }
    {
        // Rotate normal inward figure
        if (d < 0) {
            a = -a;
            b = -b;
            c = -c;
            d = -d;
        }
    }

    [[nodiscard]] __host__ __device__ bool isOnGoodSide(Point p) const { return a * p.x + b * p.y + c * p.z + d >= 0; }

private:
    const float3 DETERMINANTS;

    float a;
    float b;
    float c;
    float d;
};

constexpr float CUBOID_VOLUME = (MAX_X - MIN_X) * (MAX_Y - MIN_Y) * (MAX_Z - MIN_Z);

__device__ constexpr Plane ABC { A, B, C };
__device__ constexpr Plane ABD { A, B, D };
__device__ constexpr Plane BCD { B, C, D };
__device__ constexpr Plane CAD { C, A, D };

class ProcessorType { };
class CPU : ProcessorType { };
class GPU : ProcessorType { };

template <typename ProcessorType> float computeVolume();

template <> float computeVolume<CPU>()
{
    random_device rd;
    mt19937 generator { rd() };
    uniform_real_distribution<float> xDistribution { min({ A.x, B.x, C.x, D.x }), max({ A.x, B.x, C.x, D.x }) };
    uniform_real_distribution<float> yDistribution { min({ A.y, B.y, C.y, D.y }), max({ A.y, B.y, C.y, D.y }) };
    uniform_real_distribution<float> zDistribution { min({ A.z, B.z, C.z, D.z }), max({ A.z, B.z, C.z, D.z }) };

    int innerPoints = 0;
    for (int i = 0; i < TOTAL_ITERATIONS; ++i) {
        const Point point { xDistribution(generator), yDistribution(generator), zDistribution(generator) };
        if (ABC.isOnGoodSide(point) and ABD.isOnGoodSide(point) and BCD.isOnGoodSide(point)
            and CAD.isOnGoodSide(point)) {
            ++innerPoints;
        }
    }
    return CUBOID_VOLUME * static_cast<float>(innerPoints) / TOTAL_ITERATIONS;
}

__host__ __device__ float adjustRandomNumber(float value, float min, float max) { return value * (max - min) + min; }

__managed__ unsigned long long innerPointsAccumulator = 0;

__global__ void accumulateInnerPoints()
{
    const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= TOTAL_USED_THREADS) {
        return;
    }

    constexpr int SEED = 1337;
    curandState randomState;
    curand_init(SEED, threadId, 0, &randomState);

    int innerPoints = 0;
    for (int i = 0; i < ITERATIONS_PER_THREAD; ++i) {
        const Point point {
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


template <> float computeVolume<GPU>()
{
    constexpr int THREADS_PER_BLOCK = 128;
    constexpr int BLOCKS_PER_GRID = (TOTAL_USED_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    accumulateInnerPoints<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
    CUDA_ASSERT(cudaDeviceSynchronize())
    return CUBOID_VOLUME * static_cast<float>(innerPointsAccumulator) / TOTAL_ITERATIONS;
}

int main()
{
    cout << computeVolume<CPU>() << endl;
    cout << computeVolume<GPU>() << endl;
}
