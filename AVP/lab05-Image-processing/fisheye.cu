#include "fisheye.cuh"

#include "utility.cuh"

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

__global__
void transform(Pixel<>* sourceImage, const size_t sourceImagePitch,
               Pixel<>* targetImage, const size_t targetImagePitch,
               bool* interpolationMask, const size_t interpolationMaskPitch,
               const int width, const int height,
               const float coefficient) {
    const unsigned x = (threadIdx.x + blockDim.x * blockIdx.x) * PIXELS_PER_THREAD;
    const unsigned y = threadIdx.y + blockDim.y * blockIdx.y;
    if (y >= height) {
        return;
    }
    const auto* sourceRow = static_cast<Pixel<>*>(rowPitched(sourceImage, sourceImagePitch, y));
    for (int i = 0; i < PIXELS_PER_THREAD; ++i) {
        if ((x + i) >= width) {
            return;
        }
        const float2 normalized{normalizeValue(static_cast<float>(x + i), static_cast<float>(width)),
                                normalizeValue(static_cast<float>(y), static_cast<float>(height))};

        const float radius = sqrt(normalized.x * normalized.x + normalized.y * normalized.y);
        const float theta = atan2(normalized.y, normalized.x);

        constexpr float epsilon = 0.001f;
        const float scale = min(1.0f / abs(cos(theta) + epsilon), 1.0f / abs(sin(theta) + epsilon));

        const float newRadius = min(scale, 1.0f) * pow(radius, coefficient);
        const auto newX = static_cast<int>(round(
                (static_cast<float>(width) - 1.0f) / 2.0f * newRadius * cos(theta) +
                (static_cast<float>(width) - 1.0f) / 2.0f
        ));
        const auto newY = static_cast<int>(round(
                (static_cast<float>(height) - 1.0f) / 2.0f * newRadius * sin(theta) +
                (static_cast<float>(height) - 1.0f) / 2.0f
        ));
        auto* targetRow = static_cast<Pixel<>*>(rowPitched(targetImage, targetImagePitch, newY));
        auto* interpolationMaskRow = static_cast<bool*>(rowPitched(interpolationMask, interpolationMaskPitch, newY));
        if (0 <= newX and newX < width and
            0 <= newY and newY < height) {
            targetRow[newX] = sourceRow[x + i];
            interpolationMaskRow[newX] = true;
        }
    }
}

void fisheyeTransform(const Image& source, Image& target, MarkerImage& interpolationMask, const float coefficient) {
    const dim3 GRID_DIM{
            (source.width + BLOCK_DIM.x * PIXELS_PER_THREAD - 1) / (BLOCK_DIM.x * PIXELS_PER_THREAD),
            (source.height + BLOCK_DIM.y - 1) / BLOCK_DIM.y
    };
    transform<<<GRID_DIM, BLOCK_DIM>>>(source.deviceData, source.pitch,
                                       target.deviceData, target.pitch,
                                       interpolationMask.deviceData, interpolationMask.pitch,
                                       source.width, source.height,
                                       coefficient);
    CUDA_ASSERT(cudaMemcpy2D(target.data, target.width * sizeof(Pixel<>),
                             target.deviceData, target.pitch,
                             target.width * sizeof(Pixel<>), target.height,
                             cudaMemcpyDeviceToHost))
    CUDA_ASSERT(cudaMemcpy2D(interpolationMask.data, interpolationMask.width * sizeof(bool),
                             interpolationMask.deviceData, interpolationMask.pitch,
                             interpolationMask.width * sizeof(bool), interpolationMask.height,
                             cudaMemcpyDeviceToHost))
}
