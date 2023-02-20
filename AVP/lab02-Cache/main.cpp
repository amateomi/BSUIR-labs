#include <iostream>
#include <chrono>
#include <string_view>
#include <vector>

#include <matplot/matplot.h>

// Restrictions: OS - Linux, CPU - AMD Ryzen 4600HS

using namespace std;
using namespace chrono;
using namespace matplot;

using Latencies = vector<long>;

class Timer {
public:
    using ClockType = high_resolution_clock;

    Timer(Latencies& latencies, int n)
            : m_Latencies{latencies} {
        cout << n << ": ";
        m_StartTimePoint = ClockType::now();
    }

    ~Timer() {
        auto elapsedMilliseconds = duration_cast<milliseconds>(ClockType::now() - m_StartTimePoint).count();
        cout << elapsedMilliseconds << endl;
        m_Latencies.push_back(elapsedMilliseconds);
    }

private:
    Latencies& m_Latencies;
    time_point<ClockType> m_StartTimePoint;
};

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

void drawPlot(Latencies& latencies) {
    plot(latencies);
    title("Cache latency");
    xlabel("N");
    xrange({2, N});
    text(9, latencies[8], "8-way");
    text(17, latencies[16], "16-way");
    ylabel("Latency");
    show();
}

int main() {
    Latencies latencies{0};
    auto* array = new(align_val_t{64}) CacheLineElement[N * TOTAL_ELEMENTS_IN_CACHE];
    for (int n = 2; n < N; ++n) {
        init(array, n);
        volatile size_t t{};
        Timer timer{latencies, n};
        for (int i = 0; i < 1'000'000'000; ++i) {
            t = array[t].index;
        }
    }
    operator delete(array, align_val_t{64});
    drawPlot(latencies);
}
