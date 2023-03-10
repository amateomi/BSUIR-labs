#include <iostream>
#include <iomanip>

using namespace std;

#define CUDA_ASSERT(cudaError)                                                               \
    if (cudaError != cudaSuccess) {                                                          \
        cerr << cudaGetErrorString(cudaError) << ' ' << __FILE__ << ' ' << __LINE__ << endl; \
        exit(1);                                                                             \
    }

#define MATRIX_TYPE 0

#if MATRIX_TYPE == 0
constexpr int N = 7;
constexpr int M = 9;
#elif MATRIX_TYPE == 1
constexpr int N = 6;
constexpr int M = 4;
#elif MATRIX_TYPE == 2
constexpr int N = 5;
constexpr int M = 5;
#endif

__device__ __managed__
int source[N * M]{
#if MATRIX_TYPE == 0
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        28, 29, 30, 31, 32, 33, 34, 35, 10,
        27, 48, 49, 50, 51, 52, 53, 36, 11,
        26, 47, 60, 61, 62, 63, 54, 37, 12,
        25, 46, 59, 58, 57, 56, 55, 38, 13,
        24, 45, 44, 43, 42, 41, 40, 39, 14,
        23, 22, 21, 20, 19, 18, 17, 16, 15
#elif MATRIX_TYPE == 1
        1, 2, 3, 4,
        16, 17, 18, 5,
        15, 24, 19, 6,
        14, 23, 20, 7,
        13, 22, 21, 8,
        12, 11, 10, 9
#elif MATRIX_TYPE == 2
        1, 2, 3, 4, 5,
        16, 17, 18, 19, 6,
        15, 24, 25, 20, 7,
        14, 23, 22, 21, 8,
        13, 12, 11, 10, 9
#endif
};
__device__ __managed__
int destination[N * M];

__global__
void applyDarkMagic() {
    const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned i = id / M;
    const unsigned j = id % M;

    const unsigned depth = min(min(i, j), min(N - i - 1, M - j - 1));
    const unsigned totalUpperDepthsElements = 2 * depth * (N + M - 2 * depth);

    if (bool isTop = i <= N / 2 && i < j + 1 && j < M - i; isTop) {
        destination[totalUpperDepthsElements + id % (M + 1)] = source[id];

    } else if (bool isBottom = i >= N / 2 && N - i - 1 < j + 1 && j < M - (N - i - 1); isBottom) {
        destination[totalUpperDepthsElements + N + N * M + M - 3 - depth * (M + 5) - id] = source[id];

    } else if (bool isLeft = j < M / 2 && j < i && i < N - j - 1; isLeft) {
        destination[totalUpperDepthsElements + 2 * M + N - 6 * depth - 2 +
                    (N * M - (2 + depth) * M + depth - id) / M] = source[id];

    } else if (bool isRight = j >= M / 2 && M - j - 1 < i && i < N - (M - j); isRight) {
        destination[totalUpperDepthsElements + M - 2 * depth +
                    (id - (2 + depth) * M + 1 + depth) / M] = source[id];
    }
}

std::ostream& operator<<(std::ostream& os, const int matrix[]) {
    for (int i = 0; i < N * M; ++i) {
        if (i % M == 0 && i != 0) {
            os << endl;
        }
        os << setw(3) << matrix[i] << ' ';
    }
    return os << endl;
}

int main() {
    cout << "Source matrix:\n"
         << source << endl;

    applyDarkMagic<<<N * M + 127 / 128, 128>>>();
    CUDA_ASSERT(cudaDeviceSynchronize())

    cout << "Destination matrix:\n"
         << destination << endl;
}
