#include "interpolation.cuh"

#include "utility.cuh"

[[nodiscard]]
__device__
uint2 findTopLeftColoredPixel(bool* mask, const size_t maskPitch, const uint2 start) {
    const unsigned maxStep = min(start.x, start.y);
    for (unsigned step = 1; step < maxStep; ++step) {
        for (unsigned i = 0; i < step; ++i) {
            const unsigned x = start.x - step + i;
            const unsigned y = start.y - i - 1;
            const auto* maskRow = static_cast<bool*>(rowPitched(mask, maskPitch, y));
            if (maskRow[x]) {
                return {x, y};
            }
        }
    }
    return {0, 0};
}

[[nodiscard]]
__device__
uint2 findTopRightColoredPixel(bool* mask, const size_t maskPitch, const int width, const uint2 start) {
    const unsigned maxStep = min(width - start.x, start.y);
    for (unsigned step = 1; step < maxStep; ++step) {
        for (unsigned i = 0; i < step; ++i) {
            const unsigned x = start.x + step - i;
            const unsigned y = start.y - i - 1;
            const auto* maskRow = static_cast<bool*>(rowPitched(mask, maskPitch, y));
            if (maskRow[x]) {
                return {x, y};
            }
        }
    }
    return {static_cast<unsigned>(width - 1), 0};
}

[[nodiscard]]
__device__
uint2 findBottomLeftColoredPixel(bool* mask, const size_t maskPitch, const int height, const uint2 start) {
    const unsigned maxStep = min(start.x, height - start.y);
    for (unsigned step = 1; step < maxStep; ++step) {
        for (unsigned i = 0; i < step; ++i) {
            const unsigned x = start.x - step + i;
            const unsigned y = start.y + i + 1;
            const auto* maskRow = static_cast<bool*>(rowPitched(mask, maskPitch, y));
            if (maskRow[x]) {
                return {x, y};
            }
        }
    }
    return {0, static_cast<unsigned>(height - 1)};
}

[[nodiscard]]
__device__
uint2 findBottomRightColoredPixel(bool* mask, const size_t maskPitch,
                                  const int width, const int height, const uint2 start) {
    const unsigned maxStep = min(width - start.x, height - start.y);
    for (unsigned step = 1; step < maxStep; ++step) {
        for (int i = 0; i < step; ++i) {
            const unsigned x = start.x + step - i;
            const unsigned y = start.y + i + 1;
            const auto* maskRow = static_cast<bool*>(rowPitched(mask, maskPitch, y));
            if (maskRow[x]) {
                return {x, y};
            }
        }
    }
    return {static_cast<unsigned>(width - 1), static_cast<unsigned>(height - 1)};
}

[[nodiscard]]
__device__
inline float calculateScale(const float min, const float max, const unsigned value) {
    return (max - static_cast<float>(value)) / (max - min);
}

[[nodiscard]]
__device__
inline float calculateScale(const unsigned min, const unsigned max, const unsigned value) {
    return calculateScale(static_cast<float>(min), static_cast<float>(max), value);
}

[[nodiscard]]
__device__
inline float bilinearInterpolation(const float scale, const float firstColor, const float secondColor) {
    return scale * firstColor + (1.0f - scale) * secondColor;
}

[[nodiscard]]
__device__
inline float bilinearInterpolation(const float scale, const uint8_t firstColor, const uint8_t secondColor) {
    return bilinearInterpolation(scale, static_cast<float>(firstColor), static_cast<float>(secondColor));
}

[[nodiscard]]
__device__
inline Pixel<float> calculatePixelColor(const float scale, const Pixel<> firstPixel, const Pixel<> secondPixel) {
    return {bilinearInterpolation(scale, firstPixel.red, secondPixel.red),
            bilinearInterpolation(scale, firstPixel.green, secondPixel.green),
            bilinearInterpolation(scale, firstPixel.blue, secondPixel.blue),
    };
}

[[nodiscard]]
__device__
inline Pixel<> calculatePixelColor(const float scale, const Pixel<float> firstPixel, const Pixel<float> secondPixel) {
    return {static_cast<uint8_t>(bilinearInterpolation(scale, firstPixel.red, secondPixel.red)),
            static_cast<uint8_t>(bilinearInterpolation(scale, firstPixel.green, secondPixel.green)),
            static_cast<uint8_t>(bilinearInterpolation(scale, firstPixel.blue, secondPixel.blue)),
    };
}

[[nodiscard]]
__device__
inline float linearInterpolation(const uint2 point0, const uint2 point1, const unsigned x) {
    return (static_cast<float>(point0.y) * static_cast<float>(point1.x - x) +
            static_cast<float>(point1.y) * static_cast<float>(x - point0.x)) /
           static_cast<float>(point1.x - point0.x);
}

__global__
void interpolate(bool* mask, const size_t maskPitch,
                 Pixel<>* targetImage, const size_t targetImagePitch,
                 const int width, const int height) {
    const unsigned x = (threadIdx.x + blockDim.x * blockIdx.x) * PIXELS_PER_THREAD;
    const unsigned y = threadIdx.y + blockDim.y * blockIdx.y;
    if (y >= height) {
        return;
    }
    const auto* maskRow = static_cast<bool*>(rowPitched(mask, maskPitch, y));
    auto* targetRow = static_cast<Pixel<>*>(rowPitched(targetImage, targetImagePitch, y));
    for (int i = 0; i < PIXELS_PER_THREAD; ++i) {
        if ((x + i) >= width) {
            return;
        }
        if (maskRow[x + i]) {
            continue;
        }
        const uint2 topLeft{findTopLeftColoredPixel(mask, maskPitch, {x + i, y})};
        const uint2 topRight{findTopRightColoredPixel(mask, maskPitch, width, {x + i, y})};
        const uint2 bottomLeft{findBottomLeftColoredPixel(mask, maskPitch, height, {x + i, y})};
        const uint2 bottomRight{findBottomRightColoredPixel(mask, maskPitch, width, height, {x + i, y})};

        const auto topLeftPixel = static_cast<Pixel<>*>(
                rowPitched(targetImage, targetImagePitch, topLeft.y)
        )[topLeft.x];
        const auto topRightPixel = static_cast<Pixel<>*>(
                rowPitched(targetImage, targetImagePitch, topRight.y)
        )[topRight.x];
        const auto bottomLeftPixel = static_cast<Pixel<>*>(
                rowPitched(targetImage, targetImagePitch, bottomLeft.y)
        )[bottomLeft.x];
        const auto bottomRightPixel = static_cast<Pixel<>*>(
                rowPitched(targetImage, targetImagePitch, bottomRight.y)
        )[bottomRight.x];

        const float topScale = calculateScale(topLeft.x, topRight.x, x + i);
        const auto topPixel = calculatePixelColor(topScale, topLeftPixel, topRightPixel);

        const float bottomScale = calculateScale(bottomLeft.x, bottomRight.x, x + i);
        const auto bottomPixel = calculatePixelColor(bottomScale, bottomLeftPixel, bottomRightPixel);

        const float topY = linearInterpolation(topLeft, topRight, x + i);
        const float bottomY = linearInterpolation(bottomLeft, bottomRight, x + i);
        const float scale = calculateScale(topY, bottomY, y);

        targetRow[x + i] = calculatePixelColor(scale, topPixel, bottomPixel);
    }
}

void interpolate(const MarkerImage& mask, Image& target) {
    const dim3 GRID_DIM{
            (target.width + BLOCK_DIM.x * PIXELS_PER_THREAD - 1) / (BLOCK_DIM.x * PIXELS_PER_THREAD),
            (target.height + BLOCK_DIM.y - 1) / BLOCK_DIM.y
    };
    interpolate<<<GRID_DIM, BLOCK_DIM>>>(mask.deviceData, mask.pitch,
                                         target.deviceData, target.pitch,
                                         target.width, target.height);
}
