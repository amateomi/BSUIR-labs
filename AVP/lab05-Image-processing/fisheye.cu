#include "fisheye.cuh"

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
