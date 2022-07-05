#include "MatrixSize.hpp"

//----------------------Constructors & Destructors-----------------------//

MatrixSize::MatrixSize(MatrixSize::String string, MatrixSize::Column column) {
    this->stringTotal = string;
    this->columnTotal = column;
}

//-------------------------------Overloads-------------------------------//

std::ostream &operator<<(std::ostream &out, const MatrixSize &matrixSize) {
    out << "Matrix " << matrixSize.stringTotal << 'x' << matrixSize.columnTotal;

    return out;
}
