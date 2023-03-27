#pragma once

#include <algorithm>

using namespace std;

using Point = float3;
using Vector = float3;

constexpr Point A{0, 0, -1};
constexpr Point B{-1, 0, 0};
constexpr Point C{0, -1, 0};
constexpr Point D{1, 1, 1};

constexpr float MIN_X = min({A.x, B.x, C.x, D.x});
constexpr float MAX_X = max({A.x, B.x, C.x, D.x});

constexpr float MIN_Y = min({A.y, B.y, C.y, D.y});
constexpr float MAX_Y = max({A.y, B.y, C.y, D.y});

constexpr float MIN_Z = min({A.z, B.z, C.z, D.z});
constexpr float MAX_Z = max({A.z, B.z, C.z, D.z});

constexpr float CUBOID_VOLUME = (MAX_X - MIN_X) * (MAX_Y - MIN_Y) * (MAX_Z - MIN_Z);

[[nodiscard]]
constexpr Vector computeVector(Point p1, Point p2) {
    return {p2.x - p1.x, p2.y - p1.y, p2.z - p1.z};
}

[[nodiscard]]
constexpr float computeDeterminant(float2 row1, float2 row2) {
    return row1.x * row2.y - row1.y * row2.x;
}

[[nodiscard]]
constexpr float computeDeterminant(float3 row1, float3 row2, float3 row3) {
    return row1.x * computeDeterminant({row2.y, row2.z}, {row3.y, row3.z}) -
           row1.y * computeDeterminant({row2.x, row2.z}, {row3.x, row3.z}) +
           row1.z * computeDeterminant({row2.x, row2.y}, {row3.x, row3.y});
}

struct Plane {
public:
    constexpr Plane(Point p1, Point p2, Point p3)
            : DETERMINANTS{computeDeterminant({p2.y - p1.y, p2.z - p1.z},
                                              {p3.y - p1.y, p3.z - p1.z}),
                           computeDeterminant({p2.x - p1.x, p2.z - p1.z},
                                              {p3.x - p1.x, p3.z - p1.z}),
                           computeDeterminant({p2.x - p1.x, p2.y - p1.y},
                                              {p3.x - p1.x, p3.y - p1.y})},
              a{DETERMINANTS.x},
              b{-DETERMINANTS.y},
              c{DETERMINANTS.z},
              d{-p1.x * DETERMINANTS.x + p1.y * DETERMINANTS.y - p1.z * DETERMINANTS.z} {
        // Rotate normal inward figure
        if (d < 0) {
            a = -a;
            b = -b;
            c = -c;
            d = -d;
        }
    }

    [[nodiscard]]
    __host__
    __device__
    bool isOnGoodSide(Point p) const {
        return a * p.x + b * p.y + c * p.z + d >= 0;
    }

private:
    const float3 DETERMINANTS;

    float a;
    float b;
    float c;
    float d;
};
