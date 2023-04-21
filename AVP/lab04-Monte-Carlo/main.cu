#include "computation.cuh"
#include "timers.cuh"

using namespace std;

int main() {
    {
        cudaDeviceProp deviceProperty{};
        CUDA_ASSERT(cudaGetDeviceProperties(&deviceProperty, 0))
        cout << "Device name: " << deviceProperty.name
             << "\nGlobal memory: " << static_cast<double>(deviceProperty.totalGlobalMem) / 1073741824 << " GiB"
             << "\nShared memory per block: " << static_cast<double>(deviceProperty.sharedMemPerBlock) / 1024 << " KiB"
             << "\nShared memory per multiprocessor: "
             << static_cast<double>(deviceProperty.sharedMemPerMultiprocessor) / 1024 << " KiB"
             << "\nCompute capability: " << deviceProperty.major << '.' << deviceProperty.minor << endl
             << endl;
    }
    cout << "Target volume=" << computeVolumeAccurate() << endl;
    {
        TimerCPU timer;
        const auto volume = computeVolumeCPU();
        cout << "CPU: volume=" << volume;
    }
    {
        TimerGPU timer;
        const auto volume = computeVolumeGPU();
        cout << "GPU: volume=" << volume;
    }
}
