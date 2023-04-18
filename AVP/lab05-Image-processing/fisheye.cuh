#pragma once

#include "image.cuh"
#include "marker.cuh"

[[nodiscard]]
float calculateFisheyeCoefficient(float width, float height, float radius);

void fisheyeTransform(const Image& source, Image& target, MarkerImage& interpolationMask, float coefficient);
