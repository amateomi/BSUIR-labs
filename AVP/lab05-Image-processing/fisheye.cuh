#pragma once

#include "image.cuh"
#include "marker.cuh"

[[nodiscard]]
float calculateFisheyeCoefficient(float width, float height, float radius);

void fisheyeTransform(const Image& source, Image& target, MarkerImage& interpolationMask, float coefficient);

[[nodiscard]]
__host__ __device__
inline float normalizeValue(const float value, const float max) {
    return (value - (max - 1.0f) / 2.0f) * 2.0f / (max - 1.0f);
}
