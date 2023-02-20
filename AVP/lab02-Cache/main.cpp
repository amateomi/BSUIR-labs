#include <iostream>
#include <chrono>
#include <fstream>
#include <string_view>

using namespace std;
using namespace chrono;

class Timer {
public:
    using ClockType = high_resolution_clock;

    explicit Timer(int number) {
        cout << number << ": ";
        m_TimePoint = ClockType::now();
    }

    ~Timer() {
        auto elapsed = ClockType::now() - m_TimePoint;
        cout << duration_cast<milliseconds>(elapsed).count() << endl;
    }

private:
    time_point<ClockType> m_TimePoint;
};

// Restrictions: OS - Linux, CPU - AMD Ryzen 4600HS
[[nodiscard]] size_t getL3CacheSize() {
    size_t size;
    ifstream{"/sys/devices/system/cpu/cpu0/cache/index3/size"} >> size;
    return size * 1024 * 2;
}

constexpr auto N = 30;
const auto L3_CACHE_SIZE = getL3CacheSize();
const auto TOTAL_ELEMENTS_IN_CACHE = L3_CACHE_SIZE / 64;

union CacheLineElement {
    size_t index;
    [[maybe_unused]] byte padding[64];
};

void init(CacheLineElement* data, int n) {
    const auto BLOCK_SIZE_IN_BYTES = L3_CACHE_SIZE / n;
    const auto BLOCK_SIZE_IN_ELEMENTS = BLOCK_SIZE_IN_BYTES / 64;
    for (size_t i = 0; i < BLOCK_SIZE_IN_ELEMENTS; ++i) {
        for (size_t j = 0; j < n - 1; ++j) {
            data[i + j * TOTAL_ELEMENTS_IN_CACHE].index = i + (j + 1) * TOTAL_ELEMENTS_IN_CACHE;
        }
        data[i + (n - 1) * TOTAL_ELEMENTS_IN_CACHE].index = (i == BLOCK_SIZE_IN_ELEMENTS - 1) ? 0 : i + 1;
    }
}

int main() {
    auto* array = new(align_val_t{64}) CacheLineElement[N * TOTAL_ELEMENTS_IN_CACHE];
    for (int n = 2; n < N; ++n) {
        init(array, n);
        volatile size_t t{};
        Timer timer{n};
        for (int i = 0; i < 500'000'000; ++i) {
            t = array[t].index;
        }
    }
    operator delete(array, align_val_t{64});
}
