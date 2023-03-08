#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

#define CUDA_ASSERT(cudaError)                                                               \
    if (cudaError != cudaSuccess) {                                                          \
        cerr << cudaGetErrorString(cudaError) << ' ' << __FILE__ << ' ' << __LINE__ << endl; \
        exit(1);                                                                             \
    }

//__global__
//void add(int n, const float* x, float* y) {
//    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
//    const auto stride = blockDim.x * gridDim.x;
//    for (auto i = index; i < n; i += stride) {
//        y[i] = x[i] + y[i];
//    }
//}

void transpose(const int in[], int out[],
               int n, int m) {
    for (int ii = 0; ii < n * m; ++ii) {
        const int i = ii / m;
        const int j = ii % m;

        const int depth = min(min(i, j), min(n - i - 1, m - j - 1));
        const int totalUpperDepthElements = 2 * depth * (n + m - 2 * depth);

        if (bool isTop = i <= n / 2 && i - 1 < j && j < m - i; isTop) {
            out[totalUpperDepthElements + ii % (m + 1)] = in[ii];

        } else if (bool isBottom = i >= n / 2 && n - i - 2 < j && j < m - (n - i - 1); isBottom) {
            out[totalUpperDepthElements + n + n * m + m - 3 - depth * (m + 5) - ii] = in[ii];


        } else if (bool isLeft = j < m / 2 && j < i && i < n - j - 1; isLeft) {
            out[totalUpperDepthElements + 2 * m + n - 6 * depth - 2 +
                (n * m - (2 + depth) * m + depth - ii) / m] = in[ii];

        } else if (bool isRight = j >= m / 2 && m - j - 1 < i && i < n - (m - j); isRight) {
            out[totalUpperDepthElements + m - 2 * depth + (ii - (2 + depth) * m + 1 + depth) / m] = in[ii];
        }
    }
}

int main() {
#define MATRIX_TYPE 2

#if MATRIX_TYPE == 0
    constexpr int N = 7;
    constexpr int M = 9;
    int matrix[N * M]{
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            28, 29, 30, 31, 32, 33, 34, 35, 10,
            27, 48, 49, 50, 51, 52, 53, 36, 11,
            26, 47, 60, 61, 62, 63, 54, 37, 12,
            25, 46, 59, 58, 57, 56, 55, 38, 13,
            24, 45, 44, 43, 42, 41, 40, 39, 14,
            23, 22, 21, 20, 19, 18, 17, 16, 15
    };
#elif MATRIX_TYPE == 1
    constexpr int N = 6;
    constexpr int M = 4;
    int matrix[N * M]{
            1, 2, 3, 4,
            16, 17, 18, 5,
            15, 24, 19, 6,
            14, 23, 20, 7,
            13, 22, 21, 8,
            12, 11, 10, 9,
    };
#elif MATRIX_TYPE == 2
    constexpr int N = 5;
    constexpr int M = 5;
    int matrix[N * M]{
            1, 2, 3, 4, 5,
            16, 17, 18, 19, 6,
            15, 24, 25, 20, 7,
            14, 23, 22, 21, 8,
            13, 12, 11, 10, 9
    };
#endif

    for (int i = 0; i < N * M; ++i) {
        if (i % M == 0) {
            cout << endl;
        }
        cout << setw(3) << matrix[i] << " ";
    }
    cout << endl;

    int res[N * M]{};

    transpose(matrix, res, N, M);

    for (int i = 0; i < N * M; ++i) {
        if (i % M == 0) {
            cout << endl;
        }
        cout << setw(3) << res[i] << " ";
    }
//    constexpr size_t N = 1'000'000;
//
//    float* x, * y;
//    CUDA_ASSERT(cudaMallocManaged(&x, N * sizeof(float)))
//    CUDA_ASSERT(cudaMallocManaged(&y, N * sizeof(float)))
//
//    for (int i = 0; i < N; i++) {
//        x[i] = 1.0f;
//        y[i] = 2.0f;
//    }
//
//    constexpr size_t threadsAmount = 256;
//    constexpr size_t blocksAmount = (N + threadsAmount - 1) / threadsAmount;
//
//    add<<<blocksAmount, threadsAmount>>>(N, x, y);
//
//    CUDA_ASSERT(cudaDeviceSynchronize())
}
