#include <iostream>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

using namespace std;

struct Pixel {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};

constexpr Pixel THRESHOLD{5, 5, 5};

int main() {
    int width = 0;
    int height = 0;
    int channels = 0;
    uint8_t* image = stbi_load("../shrek_mark.jpg", &width, &height, &channels, 0);
    if (!image) {
        cout << "Failed to load Shrek picture ;(" << endl;
        return 1;
    }

    cout << width << " " << height << " " << channels << endl;

    ofstream mark_image{"../mark.pbm"};
    mark_image << "P1\n"
               << "# Mark image\n"
               << width << ' ' << height << '\n';

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const Pixel pixel{image[0 + (x + y * width) * 3],
                              image[1 + (x + y * width) * 3],
                              image[2 + (x + y * width) * 3]};
            constexpr int RED_COLOR = 255;
            const int red_diff = abs(pixel.red - RED_COLOR);
            mark_image << (red_diff <= THRESHOLD.red and
                           pixel.green <= THRESHOLD.green and
                           pixel.blue <= THRESHOLD.blue) << ' ';
        }
        mark_image << '\n';
    }
    stbi_image_free(image);
}
