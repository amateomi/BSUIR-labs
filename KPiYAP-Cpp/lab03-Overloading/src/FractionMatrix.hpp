#ifndef LAB3_SRC_FRACTIONMATRIX_HPP_
#define LAB3_SRC_FRACTIONMATRIX_HPP_

#include <initializer_list>

#include "MatrixSize.hpp"

class FractionMatrix {
    //------------------------------User Types-------------------------------//
public:
    using Height = int;
    using Width = int;
    using Fraction = double;
    using Matrix = Fraction **;

    // Use this structure to indexing elements of FractionMatrix using []
    struct MatrixIndex {
        int i{};
        int j{};
    };

    //--------------------------------Fields---------------------------------//
private:
    Height m_height;
    Width m_width;
    Matrix m_matrix;

    //----------------------Constructors & Destructors-----------------------//
public:
    explicit FractionMatrix(Height height, Width width = 1);

    ~FractionMatrix();

    //-------------------------------Overloads-------------------------------//

    // Assigment //
    // Allocate the same FractionMatrix
    FractionMatrix &operator=(const FractionMatrix &fractionMatrix);

    // Initialization //
    // Used for compile time initialization
    void operator()(const std::initializer_list<std::initializer_list<Fraction>> &list);

    // Indexing //
    // Get MatrixIndex structure to indexing specific element of FractionMatrix
    Fraction &operator[](const MatrixIndex &index);

    // Constant indexing //
    // Get MatrixIndex structure to indexing specific element of FractionMatrix
    const Fraction &operator[](const MatrixIndex &index) const;

    // Output //
    friend std::ostream &operator<<(std::ostream &out, const FractionMatrix &fractionMatrix);

    // Equality //
    // Return true if matrices have the same size and each corresponding element are equal
    bool operator==(const FractionMatrix &fractionMatrix) const;

    // Less //
    // Compare FractionMatrix with Fraction type
    // Return true if all elements in the FractionMatrix less than Fraction value
    friend bool operator<(const FractionMatrix &fractionMatrix, Fraction value);

    // Prefix increment //
    // Increment each element of FractionMatrix
    // Return reference on incremented FractionMatrix
    FractionMatrix &operator++();

    // Postfix increment //
    // Increment each element of FractionMatrix
    // Return FractionMatrix before incrementing
    FractionMatrix operator++(int);

    // Prefix decrement //
    // Decrement each element of FractionMatrix
    // Return reference on decremented FractionMatrix
    friend FractionMatrix &operator--(FractionMatrix &fractionMatrix);

    // Postfix decrement //
    // Decrement each element of FractionMatrix
    // Return FractionMatrix before decrementing
    friend FractionMatrix operator--(FractionMatrix &fractionMatrix, int);

    // Addition //
    // Get the same size FractionMatrix (use isEqualMatrixSizes before)
    // Returns a new FractionMatrix, all of whose elements are equal to the
    // sum of the corresponding elements of the operand FractionMatrices
    FractionMatrix operator+(const FractionMatrix &fractionMatrix) const;

    // Subtraction //
    // Get FractionMatrix and Fraction value
    // Returns a new FractionMatrix, all of whose elements are equal to the
    // difference of the corresponding element of the operand FractionMatrix and Fraction value
    friend FractionMatrix operator-(const FractionMatrix &fractionMatrix, Fraction value);

    // Converting to double //
    // Return average value in FractionMatrix
    explicit operator double();

    // Converting to MatrixSize //
    explicit operator MatrixSize() const;

    //-------------------------------Utilities-------------------------------//

    // Allocate memory for FractionMatrix and nullify it
    void allocateMatrix();

    // Free memory from FractionMatrix
    void deallocateMatrix();

    // Compare sizes of two FractionMatrices
    // Return true if sizes are equal
    friend bool isEqualMatrixSizes(const FractionMatrix &matrix1, const FractionMatrix &matrix2);
};

#endif //LAB3_SRC_FRACTIONMATRIX_HPP_
