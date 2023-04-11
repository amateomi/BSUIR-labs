#include <algorithm>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;
using namespace filesystem;

struct Pixel {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};

struct Image {
    explicit Image(const path& image)
            : data{reinterpret_cast<Pixel*>(stbi_load(image.c_str(), &width, &height, &channels, 0))} {
        if (!data) {
            throw runtime_error{"Failed to load picture"};
        }
    }

    Image(const int width, const int height, const int channels)
            : width{width},
              height{height},
              channels{channels},
              data{static_cast<Pixel*>(stbi__malloc(width * height * channels))} {
        if (!data) {
            throw runtime_error{"Failed to allocate memory for image"};
        }
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                data[j + i * width] = {0, 0, 0};
            }
        }
    }

    ~Image() {
        stbi_image_free(data);
    }

    [[nodiscard]]
    bool saveAsJpg(const path& fileName) const noexcept {
        return stbi_write_jpg(fileName.c_str(), width, height, channels, reinterpret_cast<uint8_t*>(data), 100);
    }

    int width{};
    int height{};
    int channels{};
    Pixel* data;
};

struct MarkerImage {
    MarkerImage(const Image& image, const Pixel markerColor, const Pixel threshold)
            : width{image.width},
              height{image.height},
              data{new bool[width * height]} {

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                const Pixel pixel{image.data[j + i * width]};
                const bool isMarked = abs(pixel.red - markerColor.red) <= threshold.red and
                                      abs(pixel.green - markerColor.green) <= threshold.green and
                                      abs(pixel.blue - markerColor.blue) <= threshold.blue;
                data[j + i * width] = isMarked;
                markedPixelsAmount += isMarked;
            }
        }

        markedPixelsCoordinates = new int2[markedPixelsAmount];
        int coordinatesIndex = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                if (data[j + i * width]) {
                    markedPixelsCoordinates[coordinatesIndex] = {j, i};
                    ++coordinatesIndex;
                }
            }
        }
    }

    ~MarkerImage() {
        delete[] data;
        delete[] markedPixelsCoordinates;
    }

    [[nodiscard]]
    bool saveAsPbm(const path& fileName) const noexcept {
        ofstream file{fileName};
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

    int width;
    int height;
    bool* data;
    int markedPixelsAmount{};
    int2* markedPixelsCoordinates{};
};

[[nodiscard]]
constexpr int calculateCircleRadius(const float x, const float y) noexcept {
    return static_cast<int>(round(sqrt(x * x + y * y)));
}

struct MarkerCircle {
    explicit MarkerCircle(const MarkerImage& markerImage) {
        const int maxRadius = min(markerImage.width, markerImage.height) / 2;
        auto* radiusAccumulator = new int[maxRadius + 1]{};

        const float2 center{static_cast<float>(markerImage.width) / 2.0f - 0.5f,
                            static_cast<float>(markerImage.height) / 2.0f - 0.5f};

        for (int i = 0; i < markerImage.markedPixelsAmount; ++i) {
            const auto pixelCoordinate = markerImage.markedPixelsCoordinates[i];
            const auto estimatedRadius = calculateCircleRadius(static_cast<float>(pixelCoordinate.x) - center.x,
                                                               static_cast<float>(pixelCoordinate.y) - center.y);
            if (estimatedRadius <= maxRadius) {
                ++radiusAccumulator[estimatedRadius];
            }
        }

        const auto* maxElement = max_element(radiusAccumulator, radiusAccumulator + maxRadius + 1);
        const auto maxElementIndex = maxElement - radiusAccumulator;
        radius = static_cast<int>(maxElementIndex);
    }

    int radius{};
};

void fisheyeTransform(const Image& source, Image& result, const float coefficient) {
    const auto M = static_cast<float>(source.height);
    const auto N = static_cast<float>(source.width);
    for (int y = 0; y < source.height; ++y) {
        for (int x = 0; x < source.width; ++x) {
            const float nx = (static_cast<float>(x) - (N + 1.0f) / 2.0f) * 2.0f / N;
            const float ny = (static_cast<float>(y) - (M + 1.0f) / 2.0f) * 2.0f / M;
        }
    }
}

int main() {
    const auto imagesDirectory = "../images/"s;
    const Image source{imagesDirectory + "shrek_with_marker.jpg"};

    constexpr Pixel markerColor{255, 0, 0};
    constexpr Pixel threshold{50, 50, 50};
    const MarkerImage markerImage{source, markerColor, threshold};
    if (!markerImage.saveAsPbm(imagesDirectory + "marker.pbm")) {
        cerr << "Failed to save markerImage" << endl;
    }

    MarkerCircle circle{markerImage};

    cout << "Circle radius is " << circle.radius << endl;
    const float fisheyeCoefficient = (log10f(static_cast<float>(min(source.width, source.height))) - 1.0f) /
                                     log10f(static_cast<float>(circle.radius));
    cout << "Fisheye coefficient is " << fisheyeCoefficient << endl;
    cout << "Result marker radius is " << pow(circle.radius, fisheyeCoefficient) << endl;

    const Image result{source.width, source.height, source.channels};
    if (!result.saveAsJpg(imagesDirectory + "result.jpg")) {
        cerr << "Failed to save result source" << endl;
    }
}
