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

template<typename T = uint8_t>
struct Pixel {
    T red;
    T green;
    T blue;
};

struct Image {
    explicit Image(const path& image)
            : data{reinterpret_cast<Pixel<>*>(stbi_load(image.c_str(), &width, &height, &channels, 0))} {
        if (!data) {
            throw runtime_error{"Failed to load " + image.string()};
        }
    }

    Image(const int width, const int height, const int channels)
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
    Pixel<>* data;
};

struct MarkerImage {
    MarkerImage(const Image& image, const Pixel<> markerColor, const Pixel<> threshold)
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

[[nodiscard]]
inline float normalizeValue(const float value, const float max) {
    return (value - (max - 1.0f) / 2.0f) * 2.0f / (max - 1.0f);
}

[[nodiscard]]
float calculateFisheyeCoefficient(const float width, const float height, const float radius) {
    const float2 normalizedSourceCirclePoint{
            normalizeValue(width / 2.0f + radius, width),
            normalizeValue(height / 2.0f, height)
    };
    const float normalizedSourceCircleRadius = sqrt(normalizedSourceCirclePoint.x * normalizedSourceCirclePoint.x +
                                                    normalizedSourceCirclePoint.y * normalizedSourceCirclePoint.y);
    const float2 normalizedTargetCirclePoint{
            normalizeValue(width / 2.0f + min(width, height) * 0.1f, width),
            normalizeValue(height / 2.0f, height)
    };
    const float normalizedTargetCircleRadius = sqrt(normalizedTargetCirclePoint.x * normalizedTargetCirclePoint.x +
                                                    normalizedTargetCirclePoint.y * normalizedTargetCirclePoint.y);
    return log(normalizedTargetCircleRadius) / log(normalizedSourceCircleRadius);
}

void fisheyeTransform(const Image& source, Image& target, MarkerImage& interpolationMask, const float coefficient) {
    const auto height = static_cast<float>(source.height);
    const auto width = static_cast<float>(source.width);
    for (int y = 0; y < source.height; ++y) {
        for (int x = 0; x < source.width; ++x) {
            const float2 normalized{normalizeValue(static_cast<float>(x), width),
                                    normalizeValue(static_cast<float>(y), height)};
            const float radius = sqrt(normalized.x * normalized.x + normalized.y * normalized.y);
            const float theta = atan2(normalized.y, normalized.x);
            constexpr float epsilon = 0.001f;
            const float scale = min(1.0f / abs(cos(theta) + epsilon), 1.0f / abs(sin(theta) + epsilon));
            const float newRadius = min(scale, 1.0f) * pow(radius, coefficient);
            const auto newX = static_cast<int>(round(
                    (width - 1.0f) / 2.0f * newRadius * cos(theta) + (width - 1.0f) / 2.0f
            ));
            const auto newY = static_cast<int>(round(
                    (height - 1.0f) / 2.0f * newRadius * sin(theta) + (height - 1.0f) / 2.0f
            ));
            if (0 <= newX and newX < source.width and
                0 <= newY and newY < source.height) {
                target.data[newX + newY * target.width] = source.data[x + y * source.width];
                interpolationMask.data[newX + newY * interpolationMask.width] = true;
            }
        }
    }
}

[[nodiscard]]
int2 findTopLeftColoredPixel(const MarkerImage& mask, const int2 start) {
    const int maxStep = min(start.x, start.y);
    for (int step = 1; step < maxStep; ++step) {
        for (int i = 0; i < step; ++i) {
            const int x = start.x - step + i;
            const int y = start.y - i - 1;
            if (mask.data[x + y * mask.width]) {
                return {x, y};
            }
        }
    }
    return {0, 0};
}

[[nodiscard]]
int2 findTopRightColoredPixel(const MarkerImage& mask, const int2 start) {
    const int maxStep = min(mask.width - start.x, start.y);
    for (int step = 1; step < maxStep; ++step) {
        for (int i = 0; i < step; ++i) {
            const int x = start.x + step - i;
            const int y = start.y - i - 1;
            if (mask.data[x + y * mask.width]) {
                return {x, y};
            }
        }
    }
    return {mask.width - 1, 0};
}

[[nodiscard]]
int2 findBottomLeftColoredPixel(const MarkerImage& mask, const int2 start) {
    const int maxStep = min(start.x, mask.height - start.y);
    for (int step = 1; step < maxStep; ++step) {
        for (int i = 0; i < step; ++i) {
            const int x = start.x - step + i;
            const int y = start.y + i + 1;
            if (mask.data[x + y * mask.width]) {
                return {x, y};
            }
        }
    }
    return {0, mask.height - 1};
}

[[nodiscard]]
int2 findBottomRightColoredPixel(const MarkerImage& mask, const int2 start) {
    const int maxStep = min(mask.width - start.x, mask.height - start.y);
    for (int step = 1; step < maxStep; ++step) {
        for (int i = 0; i < step; ++i) {
            const int x = start.x + step - i;
            const int y = start.y + i + 1;
            if (mask.data[x + y * mask.width]) {
                return {x, y};
            }
        }
    }
    return {mask.width - 1, mask.height - 1};
}

[[nodiscard]]
inline float calculateScale(const float min, const float max, const int value) {
    return (max - static_cast<float>(value)) / (max - min);
}

[[nodiscard]]
inline float calculateScale(const int min, const int max, const int value) {
    return calculateScale(static_cast<float>(min), static_cast<float>(max), value);
}

[[nodiscard]]
inline float bilinearInterpolation(const float scale, const float firstColor, const float secondColor) {
    return scale * firstColor + (1.0f - scale) * secondColor;
}

[[nodiscard]]
inline float bilinearInterpolation(const float scale, const uint8_t firstColor, const uint8_t secondColor) {
    return bilinearInterpolation(scale, static_cast<float>(firstColor), static_cast<float>(secondColor));
}

[[nodiscard]]
inline Pixel<float> calculatePixelColor(const float scale, const Pixel<> firstPixel, const Pixel<> secondPixel) {
    return {bilinearInterpolation(scale, firstPixel.red, secondPixel.red),
            bilinearInterpolation(scale, firstPixel.green, secondPixel.green),
            bilinearInterpolation(scale, firstPixel.blue, secondPixel.blue),
    };
}

[[nodiscard]]
inline Pixel<> calculatePixelColor(const float scale, const Pixel<float> firstPixel, const Pixel<float> secondPixel) {
    return {static_cast<uint8_t>(bilinearInterpolation(scale, firstPixel.red, secondPixel.red)),
            static_cast<uint8_t>(bilinearInterpolation(scale, firstPixel.green, secondPixel.green)),
            static_cast<uint8_t>(bilinearInterpolation(scale, firstPixel.blue, secondPixel.blue)),
    };
}

[[nodiscard]]
inline float linearInterpolation(const int2 point0, const int2 point1, const int x) {
    return static_cast<float>(point0.y) +
           static_cast<float>(x - point0.x) * static_cast<float>(point1.y - point0.y) /
           static_cast<float>(point1.x - point0.x);
}

void interpolate(const MarkerImage& mask, Image& target) {
    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            if (mask.data[x + y * mask.width]) {
                continue;
            }
            const int2 topLeft{findTopLeftColoredPixel(mask, {x, y})};
            const int2 topRight{findTopRightColoredPixel(mask, {x, y})};
            const int2 bottomLeft{findBottomLeftColoredPixel(mask, {x, y})};
            const int2 bottomRight{findBottomRightColoredPixel(mask, {x, y})};

            const Pixel topLeftPixel{target.data[topLeft.x + topLeft.y * target.width]};
            const Pixel topRightPixel{target.data[topRight.x + topRight.y * target.width]};
            const Pixel bottomLeftPixel{target.data[bottomLeft.x + bottomLeft.y * target.width]};
            const Pixel bottomRightPixel{target.data[bottomRight.x + bottomRight.y * target.width]};

            const float topScale = calculateScale(topLeft.x, topRight.x, x);
            const auto topPixel = calculatePixelColor(topScale, topLeftPixel, topRightPixel);

            const float bottomScale = calculateScale(bottomLeft.x, bottomRight.x, x);
            const auto bottomPixel = calculatePixelColor(bottomScale, bottomLeftPixel, bottomRightPixel);

            const float topY = linearInterpolation(topLeft, topRight, x);
            const float bottomY = linearInterpolation(bottomLeft, bottomRight, x);
            const float scale = calculateScale(topY, bottomY, y);
            target.data[x + y * target.width] = calculatePixelColor(scale, topPixel, bottomPixel);
        }
    }
}

int main() {
    const auto imagesDirectory = "../images/"s;
    const Image source{imagesDirectory + "shrek_mark.jpg"};

    constexpr Pixel<> markerColor{255, 0, 0};
    constexpr Pixel<> threshold{50, 50, 50};
    const MarkerImage markerImage{source, markerColor, threshold};
    if (!markerImage.saveAsPbm(imagesDirectory + "marker.pbm")) {
        cerr << "Failed to save marker.pbm" << endl;
    }

    MarkerCircle circle{markerImage};

    cout << "Circle radius is " << circle.radius << endl;
    cout << "Target radius is " << static_cast<float>(min(markerImage.width, markerImage.height)) * 0.1f << endl;
    const float fisheyeCoefficient = calculateFisheyeCoefficient(static_cast<float>(markerImage.width),
                                                                 static_cast<float>(markerImage.height),
                                                                 static_cast<float>(circle.radius));
    cout << "Fisheye coefficient is " << fisheyeCoefficient << endl;

    Image fisheye{source.width, source.height, source.channels};
    MarkerImage interpolationMask{fisheye.width, source.height};
    fisheyeTransform(source, fisheye, interpolationMask, fisheyeCoefficient);
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
