#include "terminate-and-unexpected.hpp"

#include <iostream>

void terminate() {
    std::cout << "terminate called";
    std::abort();
}

void unexpected() {
    std::cout << "unexpected called" << std::endl;
    std::cout << "throw 123" << std::endl;
    throw 123;
}

void fun() throw(int) {
    std::cout << "throw runtime error" << std::endl;
    throw std::runtime_error("runtime error");
}