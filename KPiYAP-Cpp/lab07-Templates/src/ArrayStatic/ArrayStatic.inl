//----------------------------Statics and consts-----------------------------//

template<class T, std::size_t maxSize>
constexpr std::size_t ArrayStatic<T, maxSize>::getMaxArraySize() {
    return maxSize;
}

//------------------------Constructors & Destructors-------------------------//

template<class T, std::size_t maxSize>
void ArrayStatic<T, maxSize>::erase() {
    for (auto &item: array) {
        item = 0;
    }
}

//---------------------------------Overloads---------------------------------//

template<class T, std::size_t maxSize>
T &ArrayStatic<T, maxSize>::operator[](std::size_t index) {
    assert(0 <= index && index < maxSize);
    return array[index];
}

template<class T, std::size_t maxSize>
const T &ArrayStatic<T, maxSize>::operator[](std::size_t index) const {
    assert(0 <= index && index < maxSize);
    return array[index];
}

//-----------------------------------Output----------------------------------//

template<class T, std::size_t maxSize>
void ArrayStatic<T, maxSize>::print() const {
    for (auto &item: array) {
        std::cout << item << ' ';
    }
    std::cout << std::endl;
}


