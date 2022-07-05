#include "MosinRifle/MosinRifle.hpp"

int main() {
    MosinRifle mosinRifle{3,
            300.55, 3.5,
            2.1, "steel",
            MosinRifle::ModificationType::Combat};

    std::cout << std::endl;
    mosinRifle.show();
    std::cout << std::endl;

    return 0;
}
