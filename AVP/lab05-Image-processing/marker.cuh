#pragma once

#include "image.cuh"

struct MarkerImage {
    MarkerImage(const Image& image, Pixel<> markerColor, Pixel<> threshold);

    MarkerImage(int width, int height);

    ~MarkerImage();

    // Return false on failure
    [[nodiscard]]
    bool saveAsPbm(string_view fileName) const;

    int width;
    int height;
    bool* data;

    [[maybe_unused]] int markedPixelsAmount{};
    [[maybe_unused]] int2* markedPixelsCoordinates{};
};

struct MarkerCircle {
    explicit MarkerCircle(const MarkerImage& markerImage);

    int radius{};
};

[[nodiscard]]
inline int calculateCircleRadius(const float x, const float y) {
    return static_cast<int>(round(sqrt(x * x + y * y)));
}
