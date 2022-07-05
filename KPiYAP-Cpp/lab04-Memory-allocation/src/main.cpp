#include "CharArray3D.hpp"

#include <iostream>

int main() {
    //----------------------------Additional Task----------------------------//
    // Create new object
    CharArray3D::Dimentions dimensions{2, 2, 3};
    CharArray3D array3D{dimensions};

    // Input necessary data
    CharArray3D::Index index;
    for (index.i = 0; index.i < dimensions.i; ++index.i) {
        for (index.j = 0; index.j < dimensions.j; ++index.j) {
            for (index.k = 0; index.k < dimensions.k; ++index.k) {
                std::cout << "Enter array3D[" << index.i << "][" << index.j << "][" << index.k << "]: ";
                std::cin >> array3D[index];
            }
        }
    }
    std::cout << std::endl;

    std::cout << "Entered array3D:\n" << array3D << std::endl;

    std::cout << "Dynamic creating an object using an initializer_list:\n";
    auto *array1 = new CharArray3D{
            {
                    {
                            {'9', '8', '7'}, {'6', 'Z', '4'}
                    },
                    {
                            {'3', 'A', '1'}, {'0', 'A', 'Z'}
                    }
            }
    };
    std::cout << "\narray1:\n" << *array1;

    std::cout << "\narray3D + array1:\n";
    for (index.i = 0; index.i < dimensions.i; ++index.i) {
        std::cout << "Level " << index.i << std::endl;
        for (index.j = 0; index.j < dimensions.j; ++index.j) {
            for (index.k = 0; index.k < dimensions.k; ++index.k) {
                std::cout << static_cast<char>(array3D[index] + (*array1)[index]) << ' ';
            }
            std::cout << std::endl;
        }
    }

    std::cout << "\narray3D - array1:\n";
    for (index.i = 0; index.i < dimensions.i; ++index.i) {
        std::cout << "Level " << index.i << std::endl;
        for (index.j = 0; index.j < dimensions.j; ++index.j) {
            for (index.k = 0; index.k < dimensions.k; ++index.k) {
                std::cout << static_cast<char>(array3D[index] - (*array1)[index]) << ' ';
            }
            std::cout << std::endl;
        }
    }

    //-------------------------------First task------------------------------//
    std::cout << "\nDynamic creating an object using the copy constructor:\n";
    auto *array2 = new CharArray3D{*array1};

    std::cout << "\nBefore sorting:\n";
    std::cout << "\narray1:\n" << *array1 << "\narray2:\n" << *array2 << std::endl;

    array1->sortSelection();

    std::cout << "After sorting first object:\n";
    std::cout << "\narray1:\n" << *array1 << "\narray2:\n" << *array2 << std::endl;

    std::cout << "Deleting first object:\n";
    delete array1;
    std::cout << "\nDeleting second object:\n";
    delete array2;

    //------------------------------Second task------------------------------//
    std::cout << "\nDynamic creating an array4D using an initializer_list:\n";
    auto *array4D = new CharArray3D[2]{
            {
                    {
                            {
                                    {'1', '2'},
                                    {'5', '6'}
                            },
                            {
                                    {'3', '4'},
                                    {'7', '8'}
                            }
                    }
            },
            {
                    {
                            {
                                    {'q', 'w', 'e'},
                                    {'a', 's', 'd'}
                            },
                            {
                                    {'z', 'x', 'c'},
                                    {'v', 'b', 'n'}
                            },
                            {
                                    {'0', '9', '8'},
                                    {'7', '6', '5'}
                            }
                    }
            }
    };

    std::cout << "\narray4D[0]:\n" << array4D[0] << "\narray4D[1]\n" << array4D[1];

    std::cout << "\narray4D[0][1][1][0]:" << (array4D[0])[CharArray3D::Index{1, 1, 0}]
            << "\narray4D[1][2][1][2]:" << (array4D[1])[CharArray3D::Index{2, 1, 2}];

    std::cout << "\n\nDeleting array4D:\n";
    delete[] array4D;

    return 0;
}