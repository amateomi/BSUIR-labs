#include "marker.cuh"

#include <algorithm>
#include <fstream>

#include "utility.cuh"

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
    const auto* sourceRow = reinterpret_cast<Pixel<>*>(rowPitched(sourceImage, sourceImagePitch, y));
    auto* markerRow = reinterpret_cast<bool*>(rowPitched(markerImage, markerImagePitch, y));
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
          height{image.height},
          data{new bool[width * height]{}} {

    CUDA_ASSERT(cudaMallocPitch(&deviceData, &pitch, width * sizeof(bool), height))

    constexpr dim3 BLOCK_DIM{16, 16};
    const dim3 GRID_DIM{
            (image.width + BLOCK_DIM.x * PIXELS_PER_THREAD - 1) / (BLOCK_DIM.x * PIXELS_PER_THREAD),
            (image.height + BLOCK_DIM.y - 1) / BLOCK_DIM.y
    };

    findMarker<<<GRID_DIM, BLOCK_DIM>>>(image.deviceData, image.pitch,
                                        deviceData, pitch,
                                        width, height,
                                        markerColor, threshold);
    CUDA_ASSERT(cudaMemcpy2D(data, width * sizeof(bool),
                             deviceData, pitch,
                             width * sizeof(bool), height,
                             cudaMemcpyDeviceToHost))
}

MarkerImage::MarkerImage(const int width, const int height)
        : width{width},
          height{height},
          data{new bool[width * height]{}} {}

MarkerImage::~MarkerImage() {
    CUDA_ASSERT(cudaFree(deviceData))
    delete[] data;
}

bool MarkerImage::saveAsPbm(const string_view fileName) const {
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
    auto* row = reinterpret_cast<bool*>(rowPitched(markerImage, markerImagePitch, y));
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

MarkerCircle::MarkerCircle(const MarkerImage& markerImage) {
    const int maxRadius = min(markerImage.width, markerImage.height) / 2;
    const float2 center{static_cast<float>(markerImage.width) / 2.0f - 0.5f,
                        static_cast<float>(markerImage.height) / 2.0f - 0.5f};

    int* radiusAccumulator;
    CUDA_ASSERT(cudaMallocManaged(&radiusAccumulator, (maxRadius + 1) * sizeof(int)))

    constexpr dim3 BLOCK_DIM{16, 16};
    const dim3 GRID_DIM{
            (markerImage.width + BLOCK_DIM.x * PIXELS_PER_THREAD - 1) / (BLOCK_DIM.x * PIXELS_PER_THREAD),
            (markerImage.height + BLOCK_DIM.y - 1) / BLOCK_DIM.y
    };
    findRadius<<<GRID_DIM, BLOCK_DIM>>>(markerImage.deviceData, markerImage.pitch,
                                        markerImage.width, markerImage.height,
                                        radiusAccumulator, maxRadius, center);
    cudaDeviceSynchronize();

    const auto* maxElement = max_element(radiusAccumulator, radiusAccumulator + maxRadius + 1);
    const auto maxElementIndex = maxElement - radiusAccumulator;
    radius = static_cast<int>(maxElementIndex);
}
