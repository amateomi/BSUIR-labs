#include "FractionMatrix.hpp"

#include <cassert>

#include <iostream>
#include <iomanip>

//----------------------Constructors & Destructors-----------------------//

FractionMatrix::FractionMatrix(Height height, Width width) {
    assert(height > 0 && "Height mast be positive integer!");
    m_height = height;

    assert(width > 0 && "Height mast be positive integer!");
    m_width = width;

    m_matrix = nullptr;
    allocateMatrix();
}

FractionMatrix::~FractionMatrix() {
    deallocateMatrix();
}

//-------------------------------Overloads-------------------------------//

FractionMatrix &FractionMatrix::operator=(const FractionMatrix &fractionMatrix) {
    if (this == &fractionMatrix) {
        return *this;
    }

    deallocateMatrix();

    m_height = fractionMatrix.m_height;
    m_width = fractionMatrix.m_width;

    allocateMatrix();

    for (int i = 0; i < m_height; ++i) {
        for (int j = 0; j < m_width; ++j) {
            m_matrix[i][j] = fractionMatrix.m_matrix[i][j];
        }
    }

    return *this;
}

void FractionMatrix::operator()(const std::initializer_list<std::initializer_list<Fraction>> &list) {
    assert(list.size() == m_height && "Invalid initialize list size, not equals to height!");
    for (auto &element: list) {
        assert(element.size() == m_width && "Invalid initialize list size, not equals to width!");
    }

    // Use foreach because of initializer_list
    MatrixIndex index;
    for (auto &row: list) {
        for (auto &element: row) {
            (*this)[index] = element;
            index.j++;
        }
        index.j = 0;
        index.i++;
    }
}

FractionMatrix::Fraction &FractionMatrix::operator[](const MatrixIndex &index) {
    assert(index.i >= 0 && index.i < m_height && "Used invalid index!");
    assert(index.j >= 0 && index.j < m_width && "Used invalid index!");

    return m_matrix[index.i][index.j];
}

const FractionMatrix::Fraction &FractionMatrix::operator[](const FractionMatrix::MatrixIndex &index) const {
    assert(index.i >= 0 && index.i < m_height && "Used invalid index!");
    assert(index.j >= 0 && index.j < m_width && "Used invalid index!");

    return m_matrix[index.i][index.j];
}

std::ostream &operator<<(std::ostream &out, const FractionMatrix &fractionMatrix) {
    out.precision(4);

    FractionMatrix::MatrixIndex index;
    for (index.i = 0; index.i < fractionMatrix.m_height; ++index.i) {
        for (index.j = 0; index.j < fractionMatrix.m_width; ++index.j) {
            out << std::fixed << std::setw(10) << fractionMatrix[index];
        }
        out << std::endl;
    }

    out.precision(-1);
    return out;
}

bool FractionMatrix::operator==(const FractionMatrix &fractionMatrix) const {
    if (!isEqualMatrixSizes(*this, fractionMatrix)) {
        return false;
    }

    MatrixIndex index;
    for (index.i = 0; index.i < m_height; ++index.i) {
        for (index.j = 0; index.j < m_width; ++index.j) {
            if ((*this)[index] != fractionMatrix[index]) {
                return false;
            }
        }
    }

    return true;
}

bool operator<(const FractionMatrix &fractionMatrix, FractionMatrix::Fraction value) {
    for (int i = 0; i < fractionMatrix.m_height; ++i) {
        for (int j = 0; j < fractionMatrix.m_width; ++j) {
            if (fractionMatrix.m_matrix[i][j] >= value) {
                return false;
            }
        }
    }

    return true;
}

FractionMatrix &FractionMatrix::operator++() {
    for (int i = 0; i < m_height; ++i) {
        for (int j = 0; j < m_width; ++j) {
            ++m_matrix[i][j];
        }
    }

    return *this;
}

FractionMatrix FractionMatrix::operator++(int) {
    FractionMatrix beforeIncrement{m_height, m_width};

    beforeIncrement = *this;

    ++(*this);

    return beforeIncrement;
}

FractionMatrix &operator--(FractionMatrix &fractionMatrix) {
    for (int i = 0; i < fractionMatrix.m_height; ++i) {
        for (int j = 0; j < fractionMatrix.m_width; ++j) {
            --fractionMatrix.m_matrix[i][j];
        }
    }

    return fractionMatrix;
}

FractionMatrix operator--(FractionMatrix &fractionMatrix, int) {
    FractionMatrix beforeIncrement{fractionMatrix.m_height, fractionMatrix.m_width};

    beforeIncrement = fractionMatrix;

    --(fractionMatrix);

    return beforeIncrement;
}

FractionMatrix FractionMatrix::operator+(const FractionMatrix &fractionMatrix) const {
    assert(isEqualMatrixSizes(*this, fractionMatrix));

    FractionMatrix newMatrix{m_height, m_width};

    MatrixIndex index;
    for (index.i = 0; index.i < m_height; ++index.i) {
        for (index.j = 0; index.j < m_width; ++index.j) {
            newMatrix[index] = (*this)[index] + fractionMatrix[index];
        }
    }

    return newMatrix;
}

FractionMatrix operator-(const FractionMatrix &fractionMatrix, FractionMatrix::Fraction value) {
    FractionMatrix newMatrix{fractionMatrix.m_height, fractionMatrix.m_width};

    FractionMatrix::MatrixIndex index;
    for (index.i = 0; index.i < fractionMatrix.m_height; ++index.i) {
        for (index.j = 0; index.j < fractionMatrix.m_width; ++index.j) {
            newMatrix[index] = fractionMatrix[index] - value;
        }
    }

    return newMatrix;
}

FractionMatrix::operator double() {
    Fraction average{0};

    int countElements = 0;
    for (int i = 0; i < m_height; ++i) {
        for (int j = 0; j < m_width; ++j) {
            average += m_matrix[i][j];
            ++countElements;
        }
    }

    return average / countElements;
}

FractionMatrix::operator MatrixSize() const {
    MatrixSize matrixSize{
        static_cast<MatrixSize::String>(m_height),
        static_cast<MatrixSize::Column>(m_width)
    };

    return matrixSize;
}

//-------------------------------Utilities-------------------------------//

void FractionMatrix::allocateMatrix() {
    m_matrix = new Fraction *[m_height];
    for (int i = 0; i < m_height; ++i) {
        m_matrix[i] = new Fraction[m_width];

        for (int j = 0; j < m_width; ++j) {
            m_matrix[i][j] = 0.0;
        }
    }
}

void FractionMatrix::deallocateMatrix() {
    for (int i = 0; i < m_height; ++i) {
        delete[] m_matrix[i];
    }
    delete[] m_matrix;
}

bool isEqualMatrixSizes(const FractionMatrix &matrix1, const FractionMatrix &matrix2) {
    return matrix1.m_height == matrix2.m_height && matrix1.m_width == matrix2.m_width;
}
