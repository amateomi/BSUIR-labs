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
            throw runtime_error{"Failed to load " + image.string()};
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
              data{new bool[width * height]{}} {

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

    MarkerImage(const int width, const int height)
            : width{width},
              height{height},
              data{new bool[width * height]{}} {}

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

    [[maybe_unused]] int markedPixelsAmount{};
    [[maybe_unused]] int2* markedPixelsCoordinates{};
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

void fisheyeTransform(const Image& source, Image& target, MarkerImage& interpolationMask, const float coefficient) {
    const auto M = static_cast<float>(source.height);
    const auto N = static_cast<float>(source.width);
    for (int y = 0; y < source.height; ++y) {
        for (int x = 0; x < source.width; ++x) {
            const float nx = (static_cast<float>(x) - (N - 1.0f) / 2.0f) * 2.0f / (N - 1.0f);
            const float ny = (static_cast<float>(y) - (M - 1.0f) / 2.0f) * 2.0f / (M - 1.0f);
            const float radius = sqrt(nx * nx + ny * ny);
            const float theta = atan2(ny, nx);
            const float epsilon = 0.001f;
            const float scale = min(1.0f / abs(cos(theta) + epsilon), 1.0f / abs(sin(theta) + epsilon));
            const float newRadius = min(scale, 1.0f) * pow(radius, coefficient);
            const auto newX = static_cast<int>(round((N - 1.0f) / 2.0f * newRadius * cos(theta) + (N - 1.0f) / 2.0f));
            const auto newY = static_cast<int>(round((M - 1.0f) / 2.0f * newRadius * sin(theta) + (M - 1.0f) / 2.0f));
            if (0 <= newX and newX < source.width and
                0 <= newY and newY < source.height) {
                target.data[newX + newY * target.width] = source.data[x + y * source.width];
                interpolationMask.data[newX + newY * interpolationMask.width] = true;
            }
        }
    }
}

[[nodiscard]]
int findLeftColoredPixelX(const MarkerImage& mask, const int2 start) {
    for (int x = start.x; x >= 0; --x) {
        if (mask.data[x + start.y * mask.width]) {
            return x;
        }
    }
    return 0;
}

[[nodiscard]]
int findRightColoredPixelX(const MarkerImage& mask, const int2 start) {
    for (int x = start.x; x < mask.width; ++x) {
        if (mask.data[x + start.y * mask.width]) {
            return x;
        }
    }
    return mask.width - 1;
}

[[nodiscard]]
int findTopColoredPixelY(const MarkerImage& mask, const int2 start) {
    for (int y = start.y; y >= 0; --y) {
        if (mask.data[start.x + y * mask.width]) {
            return y;
        }
    }
    return 0;
}

[[nodiscard]]
int findBottomColoredPixelY(const MarkerImage& mask, const int2 start) {
    for (int y = start.y; y < mask.height; ++y) {
        if (mask.data[start.x + y * mask.width]) {
            return y;
        }
    }
    return mask.height - 1;
}

void interpolate(const MarkerImage& mask, Image& target) {
    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            if (mask.data[x + y * mask.width]) {
                continue;
            }
            const int leftX = findLeftColoredPixelX(mask, {x, y});
            const int rightX = findRightColoredPixelX(mask, {x, y});
            const int topY = findTopColoredPixelY(mask, {x, y});
            const int bottomY = findBottomColoredPixelY(mask, {x, y});

            const int distanceX = rightX - leftX;
            const int distanceY = bottomY - topY;

            uint8_t red;
            uint8_t green;
            uint8_t blue;
            if (distanceX < distanceY) {
                const Pixel leftPixel{target.data[leftX + y * target.width]};
                const Pixel rightPixel{target.data[rightX + y * target.width]};
                const float scale = static_cast<float>(rightX - x) / static_cast<float>(rightX - leftX);
                red = static_cast<uint8_t>(scale * static_cast<float>(leftPixel.red) +
                                           (1.0f - scale) * static_cast<float>(rightPixel.red));
                green = static_cast<uint8_t>(scale * static_cast<float>(leftPixel.green) +
                                             (1.0f - scale) * static_cast<float>(rightPixel.green));
                blue = static_cast<uint8_t>(scale * static_cast<float>(leftPixel.blue) +
                                            (1.0f - scale) * static_cast<float>(rightPixel.blue));
            } else {
                const Pixel topPixel{target.data[x + topY * target.width]};
                const Pixel bottomPixel{target.data[x + bottomY * target.width]};
                const float scale = static_cast<float>(bottomY - y) / static_cast<float>(bottomY - topY);
                red = static_cast<uint8_t>(scale * static_cast<float>(topPixel.red) +
                                           (1.0f - scale) * static_cast<float>(bottomPixel.red));
                green = static_cast<uint8_t>(scale * static_cast<float>(topPixel.green) +
                                             (1.0f - scale) * static_cast<float>(bottomPixel.green));
                blue = static_cast<uint8_t>(scale * static_cast<float>(topPixel.blue) +
                                            (1.0f - scale) * static_cast<float>(bottomPixel.blue));
            }
            target.data[x + y * target.width] = {red, green, blue};
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
        cerr << "Failed to save marker.pbm" << endl;
    }

    MarkerCircle circle{markerImage};

    cout << "Circle radius is " << circle.radius << endl;
    const float fisheyeCoefficient = (log10f(static_cast<float>(min(source.width, source.height))) - 1.0f) /
                                     log10f(static_cast<float>(circle.radius));
    cout << "Fisheye coefficient is " << fisheyeCoefficient << endl;
    cout << "Result marker radius is " << pow(circle.radius, fisheyeCoefficient) << endl;

    Image fisheye{source.width, source.height, source.channels};
    MarkerImage interpolationMask{fisheye.width, source.height};
    fisheyeTransform(source, fisheye, interpolationMask, 0.3);
    if (!fisheye.saveAsJpg(imagesDirectory + "fisheye.jpg")) {
        cerr << "Failed to save fisheye.jpg" << endl;
    }
    if (!interpolationMask.saveAsPbm(imagesDirectory + "interpolating.pbm")) {
        cerr << "Failed to save interpolating.pbm" << endl;
    }

    interpolate(interpolationMask, fisheye);
    if (!fisheye.saveAsJpg(imagesDirectory + "result.jpg")) {
        cerr << "Failed to save result.jpg" << endl;
    }
}
