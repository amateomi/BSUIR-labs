#ifndef LAB3_SRC_MATRIXSIZE_HPP_
#define LAB3_SRC_MATRIXSIZE_HPP_

#include <iostream>

class MatrixSize {
    //------------------------------User Types-------------------------------//
public:
    using String = int;
    using Column = int;

    //--------------------------------Fields---------------------------------//
private:
    String stringTotal;
    Column columnTotal;

    //----------------------Constructors & Destructors-----------------------//
public:
    explicit MatrixSize(String string, Column column);

    //-------------------------------Overloads-------------------------------//

    friend std::ostream &operator<<(std::ostream &out, const MatrixSize & matrixSize);
};


#endif //LAB3_SRC_MATRIXSIZE_HPP_
