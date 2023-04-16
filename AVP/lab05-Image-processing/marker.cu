#include "marker.cuh"

#include <algorithm>
#include <fstream>

MarkerImage::MarkerImage(const Image& image, const Pixel<> markerColor, const Pixel<> threshold)
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

MarkerImage::MarkerImage(const int width, const int height)
        : width{width},
          height{height},
          data{new bool[width * height]{}} {}

MarkerImage::~MarkerImage() {
    delete[] data;
    delete[] markedPixelsCoordinates;
}

bool MarkerImage::saveAsPbm(const string_view fileName) const {
    ofstream file{fileName.data()};
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

MarkerCircle::MarkerCircle(const MarkerImage& markerImage) {
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

