#include "image.cuh"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include "utility.cuh"

Image::Image(const string_view image)
        : data{reinterpret_cast<Pixel<>*>(stbi_load(image.data(), &width, &height, &channels, 0))} {
    if (!data) {
        throw runtime_error{"Failed to load " + string{image}};
    }
    CUDA_ASSERT(cudaMallocPitch(&deviceData, &pitch, width * sizeof(Pixel<>), height))
    CUDA_ASSERT(cudaMemcpy2D(deviceData, pitch,
                             data, width * sizeof(Pixel<>),
                             width * sizeof(Pixel<>), height,
                             cudaMemcpyHostToDevice))
}

Image::Image(const int width, const int height, const int channels)
        : width{width},
          height{height},
          channels{channels},
          data{static_cast<Pixel<>*>(stbi__malloc(width * height * channels))} {
    if (!data) {
        throw runtime_error{"Failed to allocate memory for image"};
    }
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            data[j + i * width] = {0, 0, 0};
        }
    }
}

Image::~Image() {
    CUDA_ASSERT(cudaFree(deviceData))
    stbi_image_free(data);
}

bool Image::saveAsJpg(const string_view fileName) const {
    return stbi_write_jpg(fileName.data(), width, height, channels, reinterpret_cast<uint8_t*>(data), 100);
}
