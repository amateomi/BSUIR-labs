#include "marker.cuh"

#include <algorithm>
#include <fstream>
#include <memory>

#include "utility.cuh"

[[nodiscard]]
__device__
inline int calculateCircleRadius(const float x, const float y) {
    return static_cast<int>(round(sqrt(x * x + y * y)));
}

__global__
void findMarker(Pixel<>* sourceImage, const size_t sourceImagePitch,
                bool* markerImage, const size_t markerImagePitch,
                const int width, const int height,
                const Pixel<> markerColor, const Pixel<> threshold) {
    const unsigned x = (threadIdx.x + blockDim.x * blockIdx.x) * PIXELS_PER_THREAD;
    const unsigned y = threadIdx.y + blockDim.y * blockIdx.y;
    if (y >= height) {
        return;
    }
    const auto* sourceRow = static_cast<Pixel<>*>(rowPitched(sourceImage, sourceImagePitch, y));
    auto* markerRow = static_cast<bool*>(rowPitched(markerImage, markerImagePitch, y));
    for (int i = 0; i < PIXELS_PER_THREAD; ++i) {
        if ((x + i) >= width) {
            return;
        }
        const Pixel pixel{sourceRow[x + i]};
        const bool isMarked = abs(pixel.red - markerColor.red) <= threshold.red and
                              abs(pixel.green - markerColor.green) <= threshold.green and
                              abs(pixel.blue - markerColor.blue) <= threshold.blue;
        markerRow[x + i] = isMarked;
    }
}

MarkerImage::MarkerImage(const Image& image, const Pixel<> markerColor, const Pixel<> threshold)
        : width{image.width},
          height{image.height} {
    CUDA_ASSERT(cudaMallocPitch(&deviceData, &pitch, width * sizeof(bool), height))

    const dim3 GRID_DIM{
            (image.width + BLOCK_DIM.x * PIXELS_PER_THREAD - 1) / (BLOCK_DIM.x * PIXELS_PER_THREAD),
            (image.height + BLOCK_DIM.y - 1) / BLOCK_DIM.y
    };
    findMarker<<<GRID_DIM, BLOCK_DIM>>>(image.deviceData, image.pitch,
                                        deviceData, pitch,
                                        width, height,
                                        markerColor, threshold);
}

MarkerImage::MarkerImage(const int width, const int height)
        : width{width},
          height{height} {
    CUDA_ASSERT(cudaMallocPitch(&deviceData, &pitch, width * sizeof(bool), height))
    CUDA_ASSERT(cudaMemset2D(deviceData, pitch, 0, width * sizeof(bool), height))
}

MarkerImage::~MarkerImage() noexcept(false) {
    CUDA_ASSERT(cudaFree(deviceData))
}

bool MarkerImage::saveAsPbm(const string_view fileName) const {
    auto data = make_unique<bool[]>(width * height);
    CUDA_ASSERT(cudaMemcpy2D(data.get(), width * sizeof(bool),
                             deviceData, pitch,
                             width * sizeof(bool), height,
                             cudaMemcpyDeviceToHost))
    ofstream file{fileName.data()};
    file << "P1\n"
         << "# Marker image\n"
         << width << ' ' << height << '\n';
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            file << data[j + i * width] << ' ';
        }
        file << '\n';
    }
    return file.good();
}

__global__
void findRadius(bool* markerImage, const size_t markerImagePitch,
                const int width, const int height,
                int* radiusAccumulator, const int maxRadius, const float2 center) {
    const unsigned x = (threadIdx.x + blockDim.x * blockIdx.x) * PIXELS_PER_THREAD;
    const unsigned y = threadIdx.y + blockDim.y * blockIdx.y;
    if (y >= height) {
        return;
    }
    const auto* row = reinterpret_cast<bool*>(rowPitched(markerImage, markerImagePitch, y));
    for (int i = 0; i < PIXELS_PER_THREAD; ++i) {
        if ((x + i) >= width) {
            return;
        }
        if (row[x + i]) {
            const auto estimatedRadius = calculateCircleRadius(static_cast<float>(x) - center.x,
                                                               static_cast<float>(y) - center.y);
            if (estimatedRadius <= maxRadius) {
                atomicAdd(radiusAccumulator + estimatedRadius, 1);
            }
        }
    }
}

MarkerCircle::MarkerCircle(const MarkerImage& marker) {
    const int maxRadius = min(marker.width, marker.height) / 2;
    const float2 center{static_cast<float>(marker.width) / 2.0f - 0.5f,
                        static_cast<float>(marker.height) / 2.0f - 0.5f};

    int* radiusAccumulator;
    CUDA_ASSERT(cudaMallocManaged(&radiusAccumulator, (maxRadius + 1) * sizeof(int)))
    CUDA_ASSERT(cudaMemset(radiusAccumulator, 0, (maxRadius + 1) * sizeof(int)))

    const dim3 GRID_DIM{
            (marker.width + BLOCK_DIM.x * PIXELS_PER_THREAD - 1) / (BLOCK_DIM.x * PIXELS_PER_THREAD),
            (marker.height + BLOCK_DIM.y - 1) / BLOCK_DIM.y
    };
    findRadius<<<GRID_DIM, BLOCK_DIM>>>(marker.deviceData, marker.pitch,
                                        marker.width, marker.height,
                                        radiusAccumulator, maxRadius, center);
    CUDA_ASSERT(cudaDeviceSynchronize())

    const auto* maxElement = max_element(radiusAccumulator, radiusAccumulator + maxRadius + 1);
    const auto maxElementIndex = maxElement - radiusAccumulator;
    radius = static_cast<int>(maxElementIndex);

    CUDA_ASSERT(cudaFree(radiusAccumulator))
}
