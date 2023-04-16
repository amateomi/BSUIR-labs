#include "interpolation.cuh"

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
