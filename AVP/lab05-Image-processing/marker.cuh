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

    bool* deviceData{};
    size_t pitch{};
};

struct MarkerCircle {
    explicit MarkerCircle(const MarkerImage& marker);

    int radius{};
};

[[nodiscard]]
__device__
inline int calculateCircleRadius(const float x, const float y) {
    return static_cast<int>(round(sqrt(x * x + y * y)));
}
