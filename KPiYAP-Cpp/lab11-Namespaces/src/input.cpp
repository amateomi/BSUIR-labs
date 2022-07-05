#include "input.hpp"

#include <iostream>

int InputInt() {
    int value;

    while (true) {
        try {
            std::cin >> value;
            if (std::cin.peek() != '\n') { // Catch 123zxc case
                throw std::runtime_error("Not digit in cin");
            }
        } catch (std::exception &error) {
            std::cerr << "Error: " << error.what() << std::endl;
            std::cin.clear();
            std::cin.ignore(10'000, '\n');
            continue;
        }
        break;
    }

    return value;
}

int InputPositiveInt() {
    int value;

    while (true) {
        value = InputInt();
        try {
            if (value < 0) {
                throw std::runtime_error("Value is negative");
            }
        } catch (std::exception &error) {
            std::cerr << "Error: " << error.what() << std::endl;
            continue;
        }
        break;
    }

    return value;
}

int InputPositiveIntInRange(int min, int max) {
    int value;

    while (true) {
        value = InputPositiveInt();
        try {
            if (value < min || max < value) {
                throw std::runtime_error("Value out of range");
            }
        }
        catch (std::exception &error) {
            std::cerr << "Error: " << error.what() << std::endl;
            continue;
        }
        break;
    }

    return value;
}