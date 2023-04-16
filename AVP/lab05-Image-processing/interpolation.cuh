#pragma once

#include "image.cuh"
#include "marker.cuh"

void interpolate(const MarkerImage& mask, Image& target);

[[nodiscard]]
int2 findTopLeftColoredPixel(const MarkerImage& mask, int2 start);

[[nodiscard]]
int2 findTopRightColoredPixel(const MarkerImage& mask, int2 start);

[[nodiscard]]
int2 findBottomLeftColoredPixel(const MarkerImage& mask, int2 start);

[[nodiscard]]
int2 findBottomRightColoredPixel(const MarkerImage& mask, int2 start);

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
