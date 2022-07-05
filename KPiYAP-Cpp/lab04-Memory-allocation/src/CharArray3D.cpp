#include "CharArray3D.hpp"

#include <cassert>

#include <iostream>

//-------------------------Constructors & Destructors------------------------//

CharArray3D::CharArray3D(CharArray3D::Dimentions newDimensions) {
    dimensions.i = newDimensions.i;
    dimensions.j = newDimensions.j;
    dimensions.k = newDimensions.k;

    allocate();
}

CharArray3D::CharArray3D(const std::initializer_list<std::initializer_list<std::initializer_list<char>>> &list3D) {
    dimensions.i = static_cast<Dimentions::Length>(list3D.size());
    dimensions.j = static_cast<Dimentions::Height>(list3D.begin()->size());
    dimensions.k = static_cast<Dimentions::Width>(list3D.begin()->begin()->size());

    // Check list3D sizes
    for (const auto &list2D: list3D) {
        assert(list2D.size() == dimensions.j && "Array2D: not equal heights");
        for (auto &list1D: list2D) {
            assert(list1D.size() == dimensions.k && "Array1D: not equal width");
        }
    }

    allocate();

    // Set values
    Index index;
    index.i = 0;
    for (auto &list2D: list3D) {
        index.j = 0;
        for (auto &list1D: list2D) {
            index.k = 0;
            for (auto &item: list1D) {
                (*this)[index] = item;
                ++index.k;
            }
            ++index.j;
        }
        ++index.i;
    }
}

CharArray3D::CharArray3D(const CharArray3D &toCopy) {
    dimensions = toCopy.dimensions;

    allocate();

    // Coping
    Index index;
    for (index.i = 0; index.i < dimensions.i; ++index.i) {
        for (index.j = 0; index.j < dimensions.j; ++index.j) {
            for (index.k = 0; index.k < dimensions.k; ++index.k) {
                (*this)[index] = toCopy[index];
            }
        }
    }
}

CharArray3D::~CharArray3D() {
    std::cout << "Destructor called\n";
    deallocate();
}

//---------------------------------Overloads---------------------------------//

void *CharArray3D::operator new(std::size_t size) {
    std::cout << "Operator new called\n";

    void *pointer = ::operator new(size);

    return pointer;
}

void *CharArray3D::operator new[](std::size_t size) {
    std::cout << "Operator new[] called\n";

    void *pointer = ::operator new[](size);

    return pointer;
}

void CharArray3D::operator delete(void *pointer) {
    std::cout << "Operator delete called\n";

    ::operator delete(pointer);
}

void CharArray3D::operator delete[](void *pointer) {
    std::cout << "Operator delete[] called\n";

    ::operator delete[](pointer);
}

char &CharArray3D::operator[](const CharArray3D::Index &index) {
    assert(0 <= index.i && index.i < dimensions.i && "\nIndex i invalid!\n");
    assert(0 <= index.j && index.j < dimensions.j && "\nIndex j invalid!\n");
    assert(0 <= index.k && index.k < dimensions.k && "\nIndex k invalid!\n");
    return array3D[index.i][index.j][index.k];
}

const char &CharArray3D::operator[](const CharArray3D::Index &index) const {
    return array3D[index.i][index.j][index.k];
}

std::ostream &operator<<(std::ostream &out, const CharArray3D &charArray3D) {
    for (int i = 0; i < charArray3D.dimensions.i; ++i) {
        out << "Level " << i << ":\n";
        for (int j = 0; j < charArray3D.dimensions.j; ++j) {
            for (int k = 0; k < charArray3D.dimensions.k; ++k) {
                out << charArray3D.array3D[i][j][k] << ' ';
            }
            out << std::endl;
        }
    }

    return out;
}

//---------------------------------Algorithms--------------------------------//

void CharArray3D::sortSelection() {
    Index indexCurrent;
    for (indexCurrent.i = 0; indexCurrent.i < dimensions.i; ++indexCurrent.i) {
        for (indexCurrent.j = 0; indexCurrent.j < dimensions.j; ++indexCurrent.j) {
            for (indexCurrent.k = 0; indexCurrent.k < dimensions.k; ++indexCurrent.k) {
                // Reference to current value to be swapped
                char &toSwap = (*this)[indexCurrent];
                // Pointer to the next minimal value to be swapped
                char *min = &toSwap;

                // Search next minimal element
                Index indexForSearching{indexCurrent};
                while (indexForSearching.i < dimensions.i) {
                    while (indexForSearching.j < dimensions.j) {
                        while (indexForSearching.k < dimensions.k) {
                            if ((*this)[indexForSearching] < *min) { // if current element less than found min
                                min = &(*this)[indexForSearching];
                            }
                            ++indexForSearching.k;
                        }
                        indexForSearching.k = 0; // Start search from beginning of the string
                        ++indexForSearching.j;
                    }
                    indexForSearching.j = 0; // Start search form beginning of the matrix
                    ++indexForSearching.i;
                }

                std::swap(toSwap, *min);
            }
        }
    }
}

//---------------------------------Utilities---------------------------------//

void CharArray3D::allocate() {
    array3D = new char **[dimensions.i];
    for (int i = 0; i < dimensions.i; ++i) {
        array3D[i] = new char *[dimensions.j];
        for (int j = 0; j < dimensions.j; ++j) {
            array3D[i][j] = new char[dimensions.k];
        }
    }
}

void CharArray3D::deallocate() {
    for (int i = 0; i < dimensions.i; ++i) {
        for (int j = 0; j < dimensions.j; ++j) {
            delete[] array3D[i][j];
        }
        delete[] array3D[i];
    }
    delete[] array3D;
}
