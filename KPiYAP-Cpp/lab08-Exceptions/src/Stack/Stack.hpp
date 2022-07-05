#ifndef LAB8_SRC_STACK_STACK_HPP_
#define LAB8_SRC_STACK_STACK_HPP_

#include <string>
#include <exception>

//-------------------------Stack exception handling--------------------------//

class StackException : public std::exception {
private:
    std::string errorMessage;

public:
    explicit StackException(std::string message) : errorMessage(std::move(message)) {}

    const char *what() const noexcept override {
        return errorMessage.c_str();
    }
};

// List based stack
template<class T>
class Stack {
    //--------------------------------Types----------------------------------//
private:
    // Node for list based stack
    struct StackNode {
        T data{};
        StackNode *next{};
    };

    //--------------------------------Fields---------------------------------//
private:
    // Top of the stack
    StackNode *top{};

    //----------------------Constructors & Destructors-----------------------//
public:
    Stack() = default;

    ~Stack();

    //-------------------------------Operations------------------------------//
public:
    // Add value to the top of the stack
    void push(const T &value) noexcept(false);

    // Get top value and delete it form the stack
    T pop() noexcept(false);

    // Output in cout stack data
    void print() noexcept;

    // Delete all elements from the stack
    void erase() noexcept;

    // Return amount of the elements in the stack
    std::size_t size() const noexcept;

    // Return true when stack haven't elements
    bool isEmpty() const noexcept;
};

#include "Stack.inl"

#endif //LAB8_SRC_STACK_STACK_HPP_