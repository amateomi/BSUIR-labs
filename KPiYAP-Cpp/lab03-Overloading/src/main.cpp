#include <iostream>

#include "FractionMatrix.hpp"

int main() {
    /*-----------------------------------------------------------------------*/
    // Additional task
    double a = 3.0, b = 5.0;
    std::cout << a/++b << " a = " << a << " b = " << b << std::endl
              << a/b++ << " a = " << a << " b = " << b << std::endl;
    /*-----------------------------------------------------------------------*/
    FractionMatrix m1{2, 3};

    // Initialisation
    m1({
               {1.333, 223.56,  3},
               {4,     1.0 / 3, 6},
       });

    // Assigment
    FractionMatrix m2{1, 2};
    m2 = m1;
    /*-----------------------------------------------------------------------*/
    std::cout << "After '()' and '=':\n";
    std::cout << "m1:\n" << m1 << "\nm2:\n" << m2 << std::endl;
    std::cout << "m1[1][2] = " << m1[FractionMatrix::MatrixIndex{1, 2}] << std::endl;
    /*-----------------------------------------------------------------------*/
    std::cout << "\nUnary operations:\n";

    std::cout << "++m1:\n" << ++m1 << std::endl;
    std::cout << "m1++:\n" << m1++ << "\nAfter m1++:\n" << m1 << std::endl;

    std::cout << "--m2:\n" << --m2 << std::endl;
    std::cout << "m2--:\n" << m2-- << "\nAfter m2--:\n" << m2 << std::endl;
    /*-----------------------------------------------------------------------*/
    std::cout << "Arithmetic:\n";
    std::cout << "+\n";
    std::cout << "m1:\n" << m1 << "\nm2:\n" << m2 << "\nm1 + m2\n" << m1 + m2 << std::endl;

    std::cout << "-\n";
    FractionMatrix::Fraction fraction{2.25};
    std::cout << "m1:\n" << m1 << "\nm1 - " << fraction << ":\n" << m1 - fraction << std::endl;
    /*-----------------------------------------------------------------------*/
    std::cout << "Comparing:\n";
    std::cout << "==\n";
    std::cout << "m1:\n" << m1 << "\nm2:\n" << m2 << "\nm1 == m2\n" << std::boolalpha << (m1 == m2) << std::endl;

    std::cout << "<\n";
    fraction = 255'777.12;
    std::cout << "m1:\n" << m1 << "\nm1 < " << fraction << '\n' << std::boolalpha << (m1 < fraction) << std::endl;
    /*-----------------------------------------------------------------------*/
    std::cout << "\nConverting:\n";
    std::cout << "Double (average value in matrix):\n";
    std::cout << "m2:\n" << m2 << "\nstatic_cast<double>(m2): " << static_cast<double>(m2) << std::endl;

    std::cout << "MatrixSize:\n";
    std::cout << "m2:\n" << m2 << "\nstatic_cast<MatrixSize>(m2): " << static_cast<MatrixSize>(m2) << std::endl;
    /*-----------------------------------------------------------------------*/

    return 0;
}