#include <stdio.h>

// Counts the number of digits in a natural number.
int countDigits(int number) {
    int counter = 0;

    // Algorithm: divide the number by 10 until it equals 0,
    // at each iteration we increase the counter by 1.
    do {
        number /= 10;
        counter++;
    } while (number != 0);

    return counter;
}

int main() {
    int n1 = 123;
    int n2 = 45678;

    int n1Digits = countDigits(n1);
    int n2Digits = countDigits(n2);

    if (n1Digits > n2Digits) {
        printf("n1 > n2, n1 = %i\n", n1);
    } else if (n2Digits > n1Digits) {
        printf("n2 > n1, n2 = %i\n", n2);
    } else {
        printf("n1 = n2 = %i\n", n1);
    }

    return 0;
}
