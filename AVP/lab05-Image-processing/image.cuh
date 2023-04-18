#pragma once

#include <string>
#include <string_view>
#include <stdexcept>

using namespace std;

template<typename T = uint8_t>
struct Pixel {
    T red;
    T green;
    T blue;
};

struct Image {
    explicit Image(string_view image);

    Image(int width, int height, int channels);

    ~Image() noexcept(false);

    // Return false on failure
    [[nodiscard]]
    bool saveAsJpg(string_view fileName) const;

    int width{};
    int height{};
    int channels{};

    Pixel<>* deviceData{};
    size_t pitch{};
};
