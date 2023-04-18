#pragma once

#include "image.cuh"
#include "marker.cuh"

void interpolate(const MarkerImage& mask, Image& target);
