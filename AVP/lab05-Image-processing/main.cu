#include <iostream>

#include "image.cuh"
#include "marker.cuh"
#include "fisheye.cuh"
#include "interpolation.cuh"

int main() {
    const auto imagesDirectory = "../images/"s;
    const Image source{imagesDirectory + "forest_mark.jpg"};

    constexpr Pixel<> markerColor{255, 0, 0};
    constexpr Pixel<> threshold{50, 50, 50};
    const MarkerImage markerImage{source, markerColor, threshold};
    if (!markerImage.saveAsPbm(imagesDirectory + "marker.pbm")) {
        cerr << "Failed to save marker.pbm" << endl;
    }

    MarkerCircle circle{markerImage};

    cout << "Circle radius is " << circle.radius << endl;
    cout << "Target radius is " << static_cast<float>(min(markerImage.width, markerImage.height)) * 0.1f << endl;
    const float fisheyeCoefficient = calculateFisheyeCoefficient(static_cast<float>(markerImage.width),
                                                                 static_cast<float>(markerImage.height),
                                                                 static_cast<float>(circle.radius));
    cout << "Fisheye coefficient is " << fisheyeCoefficient << endl;

    Image fisheye{source.width, source.height, source.channels};
    MarkerImage interpolationMask{fisheye.width, source.height};
    fisheyeTransform(source, fisheye, interpolationMask, fisheyeCoefficient);
    if (!fisheye.saveAsJpg(imagesDirectory + "fisheye.jpg")) {
        cerr << "Failed to save fisheye.jpg" << endl;
    }
    if (!interpolationMask.saveAsPbm(imagesDirectory + "interpolating.pbm")) {
        cerr << "Failed to save interpolating.pbm" << endl;
    }

    interpolate(interpolationMask, fisheye);
    if (!fisheye.saveAsJpg(imagesDirectory + "result.jpg")) {
        cerr << "Failed to save result.jpg" << endl;
    }
}
