#ifndef LAB4_SRC_CHARARRAY3D_HPP_
#define LAB4_SRC_CHARARRAY3D_HPP_

#include <ostream>
#include <initializer_list>

class CharArray3D {
    //-------------------------------User Types------------------------------//
public:
    struct Index {
        using Length = int;
        using Height = int;
        using Width = int;

        Length i{};
        Height j{};
        Width k{};
    };

    using Dimentions = Index;

    //---------------------------------Fields--------------------------------//
private:
    Dimentions dimensions{};
    char ***array3D{};

    //-----------------------Constructors & Destructors----------------------//
public:
    explicit CharArray3D() = default;

    // Allocate only memory for array3D
    explicit CharArray3D(Dimentions newDimensions);

    // Initialize array3D
    CharArray3D(const std::initializer_list<std::initializer_list<std::initializer_list<char>>> &list3D);

    // Deep copy
    CharArray3D(const CharArray3D &toCopy);

    ~CharArray3D();

    //-------------------------------Overloads-------------------------------//
public:
    void *operator new(std::size_t size);

    void *operator new[](std::size_t size);

    void operator delete(void *pointer);

    void operator delete[](void *pointer);

    // Indexing specific element for read/write
    char &operator[](const Index &index);

    // Indexing specific element for read only
    const char &operator[](const Index &index) const;

    // Output CharArray3D information
    friend std::ostream &operator<<(std::ostream &out, const CharArray3D &charArray3D);

    //-------------------------------Algorithms------------------------------//
public:
    // Sort chars value according to theirs ASCII codes
    void sortSelection();

    //-------------------------------Utilities-------------------------------//
private:
    void allocate();

    void deallocate();
};

#endif //LAB4_SRC_CHARARRAY3D_HPP_
