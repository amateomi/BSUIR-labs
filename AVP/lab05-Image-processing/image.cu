#include "image.cuh"

#include <memory>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include "utility.cuh"

Image::Image(const string_view image) {
    const unique_ptr<Pixel<>> data{reinterpret_cast<Pixel<>*>(
                                           stbi_load(image.data(), &width, &height, &channels, 0)
                                   )};
    if (!data) {
        throw runtime_error{"Failed to load " + string{image}};
    }
    CUDA_ASSERT(cudaMallocPitch(&deviceData, &pitch, width * sizeof(Pixel<>), height))
    CUDA_ASSERT(cudaMemcpy2D(deviceData, pitch,
                             data.get(), width * sizeof(Pixel<>),
                             width * sizeof(Pixel<>), height,
                             cudaMemcpyHostToDevice))
}

Image::Image(const int width, const int height, const int channels)
        : width{width},
          height{height},
          channels{channels} {
    CUDA_ASSERT(cudaMallocPitch(&deviceData, &pitch, width * sizeof(Pixel<>), height))
}

Image::~Image() noexcept(false) {
    CUDA_ASSERT(cudaFree(deviceData))
}

bool Image::saveAsJpg(const string_view fileName) const {
    auto data = make_unique<Pixel<>[]>(width * height);
    CUDA_ASSERT(cudaMemcpy2D(data.get(), width * sizeof(Pixel<>),
                             deviceData, pitch,
                             width * sizeof(Pixel<>), height,
                             cudaMemcpyDeviceToHost))
    return stbi_write_jpg(fileName.data(), width, height,
                          channels, reinterpret_cast<uint8_t*>(data.get()), 100);
}
