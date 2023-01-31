#include <iostream>
#include <random>
#include <chrono>
#include <string_view>
#include <functional>

#include <immintrin.h>

constexpr auto LIL_MATRIX_SIZE = 6;
constexpr auto BIG_MATRIX_SIZE = 200;

using namespace std;
using namespace chrono;

using ValueType = double;

using LilMatrix = ValueType[LIL_MATRIX_SIZE][LIL_MATRIX_SIZE];
using BigMatrix = LilMatrix[BIG_MATRIX_SIZE][BIG_MATRIX_SIZE];

[[nodiscard]] BigMatrix* createBigMatrix() {
    auto result = new BigMatrix[BIG_MATRIX_SIZE]{};
    mt19937 r{random_device{}()};
    uniform_real_distribution<ValueType> distribution{0.0, 1.0};
    for (int i = 0; i < BIG_MATRIX_SIZE; ++i) {
        for (int j = 0; j < BIG_MATRIX_SIZE; ++j) {
            for (int k = 0; k < LIL_MATRIX_SIZE; ++k) {
                for (int l = 0; l < LIL_MATRIX_SIZE; ++l) {
                    (*result)[i][j][k][l] = distribution(r);
                }
            }
        }
    }
    return result;
}

__attribute__((target("no-sse")))
void multiplyWithoutSimd(const LilMatrix& matrix1, const LilMatrix& matrix2, LilMatrix& result) {
    for (int i = 0; i < LIL_MATRIX_SIZE; ++i) {
        for (int j = 0; j < LIL_MATRIX_SIZE; ++j) {
            for (int k = 0; k < LIL_MATRIX_SIZE; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

__attribute__((target("no-sse")))
void multiplyReorder(const LilMatrix matrix1, const LilMatrix matrix2, LilMatrix result) {
    for (int i = 0; i < LIL_MATRIX_SIZE; ++i) {
        for (int k = 0; k < LIL_MATRIX_SIZE; ++k) {
            for (int j = 0; j < LIL_MATRIX_SIZE; ++j) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

__attribute__((target("sse")))
void multiplyWithSimd(const LilMatrix& matrix1, const LilMatrix& matrix2, LilMatrix& result) {
    for (int i = 0; i < LIL_MATRIX_SIZE; ++i) {
        for (int j = 0; j < LIL_MATRIX_SIZE; ++j) {
            for (int k = 0; k < LIL_MATRIX_SIZE; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

__attribute__((target("sse,avx")))
void multiplyWithMySimd(const LilMatrix& matrix1, const LilMatrix& matrix2, LilMatrix& result) {
    for (int i = 0; i < LIL_MATRIX_SIZE; ++i) {
        for (int j = 0; j < LIL_MATRIX_SIZE; ++j) {
            auto row4 = _mm256_loadu_pd(&matrix1[i][0]);
            auto row2 = _mm_loadu_pd(&matrix1[i][4]);

            auto column4 = _mm256_set_pd(matrix2[3][j],
                                         matrix2[2][j],
                                         matrix2[1][j],
                                         matrix2[0][j]);
            auto column2 = _mm_set_pd(matrix2[5][j],
                                      matrix2[4][j]);

            auto mul4 = row4 * column4;
            auto mul2 = row2 * column2;

            auto lowPart = _mm256_extractf128_pd(mul4, 0);
            auto highPart = _mm256_extractf128_pd(mul4, 1);

            auto sum4 = lowPart + highPart;
            auto sum = mul2 + sum4;

            auto value = _mm_hadd_pd(sum, sum);
            result[i][j] += _mm_cvtsd_f64(value);
        }
    }
}

using MultiplyFunction = function<void(const LilMatrix&, const LilMatrix&, LilMatrix&)>;

void calculate(const BigMatrix& matrix1,
               const BigMatrix& matrix2,
               BigMatrix& result,
               const MultiplyFunction& multiply) {

    for (int i = 0; i < BIG_MATRIX_SIZE; ++i) {
        for (int j = 0; j < BIG_MATRIX_SIZE; ++j) {
            for (int k = 0; k < BIG_MATRIX_SIZE; ++k) {
                multiply(matrix1[i][k], matrix2[k][i], result[i][j]);
            }
        }
    }
}

class Timer {
public:
    using ClockType = high_resolution_clock;

    explicit Timer(std::string_view message) {
        cout << message << ": ";
        m_TimePoint = ClockType::now();
    }

    ~Timer() {
        auto elapsed = ClockType::now() - m_TimePoint;
        cout << duration_cast<milliseconds>(elapsed).count() << "ms" << endl;
    }

private:
    time_point<ClockType> m_TimePoint;
};

[[nodiscard]] bool isEqualBigMatrices(const BigMatrix& matrix1, const BigMatrix& matrix2) {
    for (int i = 0; i < BIG_MATRIX_SIZE; ++i) {
        for (int j = 0; j < BIG_MATRIX_SIZE; ++j) {
            for (int k = 0; k < LIL_MATRIX_SIZE; ++k) {
                for (int l = 0; l < LIL_MATRIX_SIZE; ++l) {
                    constexpr ValueType epsilon = 0.001;
                    if (abs(matrix1[i][j][k][l] - matrix2[i][j][k][l]) > epsilon) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

int main() {
    const auto matrix1 = createBigMatrix();
    const auto matrix2 = createBigMatrix();

    vector<tuple<string_view, MultiplyFunction, BigMatrix*>> testCases{
            {"No SIMD",   multiplyWithoutSimd, nullptr},
            {"Reorder",   multiplyReorder,     nullptr},
            {"Auto SIMD", multiplyWithSimd,    nullptr},
            {"My AVX",    multiplyWithMySimd,  nullptr},
    };
    for (auto& [message, multiply, result]: testCases) {
        result = new BigMatrix[BIG_MATRIX_SIZE]{};
        {
            Timer timer{message};
            calculate(*matrix1, *matrix2, *result, multiply);
        }
    }

    const auto correctResult = get<BigMatrix*>(testCases.front());
    for (const auto& item: testCases) {
        const auto message = get<string_view>(item);
        const auto result = get<BigMatrix*>(item);
        std::cout << message << " is " <<
                  (isEqualBigMatrices(*correctResult, *result)
                   ? "correct" : "incorrect") << endl;
    }
}
