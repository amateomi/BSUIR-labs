#ifndef LAB7_SRC_ARRAYSTATIC_ARRAYSTATIC_HPP_
#define LAB7_SRC_ARRAYSTATIC_ARRAYSTATIC_HPP_

#include <cassert>

#include <iostream>
#include <initializer_list>

template<class T, std::size_t maxSize = 10>
class ArrayStatic {
    //--------------------------Statics and consts---------------------------//
public:
    constexpr std::size_t getMaxArraySize();
    
    //--------------------------------Fields---------------------------------//
protected:
    T array[maxSize]{};
    
    //----------------------Constructors & Destructors-----------------------//
public:
    ArrayStatic() = default;
    
    // Set all array values to zero
    virtual void erase();
    
    //-------------------------------Overloads-------------------------------//
public:
    // Read
    const T &operator[](std::size_t index) const;
    
    // Write
    T &operator[](std::size_t index);
    
    //---------------------------------Output--------------------------------//
public:
    // Display array
    virtual void print() const final;
};

#include "ArrayStatic.inl"

#endif //LAB7_SRC_ARRAYSTATIC_ARRAYSTATIC_HPP_