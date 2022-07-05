#ifndef LAB7_SRC_STACK_STACK_HPP_
#define LAB7_SRC_STACK_STACK_HPP_

#include "../ArrayStatic/ArrayStatic.hpp"

template<class T, std::size_t maxSize = 100>
class Stack : public ArrayStatic<T, maxSize> {
    //--------------------------Statics and consts---------------------------//
public:
    constexpr std::size_t getMaxStackSize();
    
    //--------------------------------Fields---------------------------------//
private:
    int topIndex{-1};
    
    //----------------------Constructors & Destructors-----------------------//
public:
    Stack() = default;
    
    // Clear stack
    void erase();
    
    //-------------------------------Operations------------------------------//
protected:

public:
    void push(const T &value);
    
    const T &pop();

    void sort();

    void find(const T &key) const;

    std::size_t size() const;

    bool isFull() const;
    
    bool isEmpty() const;
    
    //--------------------------------Overloads------------------------------//
public:
    template<std::size_t maxArgSize>
    bool operator==(const Stack<T, maxArgSize> &stack) const;
    
    template<std::size_t maxArgSize>
    bool operator!=(const Stack<T, maxArgSize> &stack) const;
    
    // Example:
    // *this: 1, 2, 3
    // stack: 4, 5, 6
    // result: 1, 2, 3, 4, 5, 6
    template<std::size_t maxArgSize>
    Stack<T, (maxSize > maxArgSize) ? maxSize : maxArgSize> operator+(const Stack<T, maxArgSize> &stack) const;
};

#include "Stack.inl"

#endif //LAB7_SRC_STACK_STACK_HPP_