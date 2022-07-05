//----------------------------Statics and consts-----------------------------//

template<class T, std::size_t maxSize>
constexpr std::size_t Stack<T, maxSize>::getMaxStackSize() {
    return maxSize;
}

//------------------------Constructors & Destructors-------------------------//

template<class T, std::size_t maxSize>
void Stack<T, maxSize>::erase() {
    topIndex = -1;
}

//---------------------------------Operations--------------------------------//

template<class T, std::size_t maxSize>
void Stack<T, maxSize>::push(const T &value) {
    assert(!isFull());
    (*this)[++topIndex] = value;
}

template<class T, std::size_t maxSize>
const T &Stack<T, maxSize>::pop() {
    assert(!isEmpty());
    return (*this)[topIndex--];
}

template<class T, std::size_t maxSize>
void Stack<T, maxSize>::sort() {
    for (int i = 0; i < size(); ++i) {
        for (int j = 0; j < size() - 1; ++j) {
            if ((*this).array[j] < (*this).array[j + 1]) {
                std::swap((*this).array[j], (*this).array[j + 1]);
            }
        }
    }
}

template<class T, std::size_t maxSize>
void Stack<T, maxSize>::find(const T &key) const {
    bool isFound = false;
    int count = 0;

    for (int i = (*this).size() - 1; i >= 0; --i) {
        if ((*this).array[i] == key) {
            std::cout << key << " is founded on " << count << " pos from head" << std::endl;
            isFound = true;
        }
        ++count;
    }

    if (!isFound) {
        std::cout << key << " not founded" << std::endl;
    }
}

template<class T, std::size_t maxSize>
std::size_t Stack<T, maxSize>::size() const {
    return topIndex + 1;
}

template<class T, std::size_t maxSize>
bool Stack<T, maxSize>::isFull() const {
    return size() == maxSize;
}

template<class T, std::size_t maxSize>
bool Stack<T, maxSize>::isEmpty() const {
    return size() == 0;
}

//----------------------------------Overloads--------------------------------//

template<class T, std::size_t maxSize>
template<std::size_t maxArgSize>
bool Stack<T, maxSize>::operator==(const Stack<T, maxArgSize> &stack) const {
    bool isEqual = true;

    if (size() == stack.size()) {
        for (int i = 0; i < size(); ++i) {
            if ((*this)[i] != stack[i]) {
                isEqual = false;
                break;
            }
        }
    } else {
        isEqual = false;
    }

    return isEqual;
}

template<class T, std::size_t maxSize>
template<std::size_t maxArgSize>
bool Stack<T, maxSize>::operator!=(const Stack<T, maxArgSize> &stack) const {
    return !operator==(stack);
}

template<class T, std::size_t maxSize>
template<std::size_t maxArgSize>
Stack<T, (maxSize > maxArgSize) ? maxSize : maxArgSize> // Return value
Stack<T, maxSize>::operator+(const Stack<T, maxArgSize> &stack) const {

    Stack<T, (maxSize > maxArgSize) ? maxSize : maxArgSize> newStack;

    assert((*this).size() + stack.size() <= newStack.getMaxStackSize());

    for (int i = 0; i < stack.size(); ++i) {
        newStack.push(stack[i]);
    }
    for (int i = 0; i < (*this).size(); ++i) {
        newStack.push((*this)[i]);
    }

    return newStack;
}

//-------------------------Template functions Output-------------------------//

template<class T, std::size_t maxSize>
void print(const Stack<T, maxSize> &stack) {
    if (stack.isEmpty()) {
        std::cout << "Stack is empty!";
    } else {
        for (int i = stack.size() - 1; i >= 0; --i) {
            std::cout << stack[i] << ' ';
        }
    }
    std::cout << std::endl;
}

// C-style strings specialisation
template<std::size_t maxSize>
void print(const Stack<char *, maxSize> &stack) {
    if (stack.isEmpty()) {
        std::cout << "Stack is empty!";
    } else {
        for (int i = stack.size() - 1; i >= 0; --i) {
            std::cout << stack[i] << std::endl;
        }
    }
    std::cout << std::endl;
}
