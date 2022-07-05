#include <iostream>

//------------------------Constructors & Destructors-------------------------//

template<class T>
Stack<T>::~Stack() {
    erase();
}

//---------------------------------Operations--------------------------------//

template<class T>
void Stack<T>::push(const T &value) {
    StackNode *newTopNode;

    // Allocation check
    try {
        newTopNode = new StackNode;
    }
    catch (std::bad_alloc &badAlloc) {
        throw StackException("Not enough memory");
    }

    // Set value to the node
    newTopNode->data = value;

    // Place node on top of the stack
    if (isEmpty()) {
        top = newTopNode;
    } else {
        newTopNode->next = top;
        top = newTopNode;
    }
}

template<class T>
T Stack<T>::pop() {
    if (isEmpty()) {
        throw StackException("Stack is empty");
    }

    // Get top value
    T value = top->data;

    // Remember top node
    StackNode *topNode = top;
    // Set second node as top
    top = top->next;
    // Deallocate old node
    delete topNode;

    return value;
}

template<class T>
void Stack<T>::print() noexcept {
    if (isEmpty()) {
        std::cout << "Stack is empty" << std::endl;
    } else {
        StackNode *currentNode = top;
        while (currentNode != nullptr) {
            std::cout << currentNode->data << " ";
            currentNode = currentNode->next;
        }
        std::cout << std::endl;
    }
}

template<class T>
void Stack<T>::erase() noexcept {
    while (!isEmpty()) {
        pop();
    }
}

template<class T>
std::size_t Stack<T>::size() const noexcept {
    std::size_t counter = 0;

    // Iterate through whole stack and count nodes
    StackNode *currentNode = top;
    while (currentNode != nullptr) {
        ++counter;
        currentNode = currentNode->next;
    }

    return counter;
}

template<class T>
bool Stack<T>::isEmpty() const noexcept {
    return (top == nullptr);
}