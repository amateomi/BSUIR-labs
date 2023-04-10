#include <algorithm>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

#include <cmath>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

using namespace std;
using namespace filesystem;

struct Pixel {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};


struct Image {
    explicit Image(const path& image)
            : data{stbi_load(image.c_str(), &width, &height, &channels, 0)} {
        if (!data) {
            throw runtime_error{"Failed to load picture"};
        }
    }

    ~Image() {
        stbi_image_free(data);
    }

    int width{};
    int height{};
    int channels{};
    uint8_t* data;
};

struct MarkerImage {
    MarkerImage(const Image& image, const Pixel markerColor, const Pixel threshold)
            : width{image.width},
              height{image.height},
              data{new bool[width * height]} {

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                const Pixel pixel{image.data[0 + (j + i * width) * 3],
                                  image.data[1 + (j + i * width) * 3],
                                  image.data[2 + (j + i * width) * 3]};
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
    bool saveAs(const path& fileName) const noexcept {
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

int main() {
    const Image image{"../images/shrek_with_marker.jpg"};
    constexpr Pixel markerColor{255, 0, 0};
    constexpr Pixel threshold{50, 50, 50};
    const MarkerImage markerImage{image, markerColor, threshold};
    if (!markerImage.saveAs("../images/marker.pbm")) {
        cerr << "Failed to save markerImage" << endl;
    }
    MarkerCircle markerCircle{markerImage};
    cout << markerCircle.radius << endl;
}
