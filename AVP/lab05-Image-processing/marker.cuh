#pragma once

#include "image.cuh"

struct MarkerImage {
    MarkerImage(const Image& image, Pixel<> markerColor, Pixel<> threshold);

    MarkerImage(int width, int height);

    ~MarkerImage() noexcept(false);

    // Return false on failure
    [[nodiscard]]
    bool saveAsPbm(string_view fileName) const;

    int width;
    int height;

    bool* deviceData{};
    size_t pitch{};
};

struct MarkerCircle {
    explicit MarkerCircle(const MarkerImage& marker);

    int radius{};
};
