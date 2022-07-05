#include <stdio.h>

int main() {
    const long long N = 100;
    const long long ARRAY_SIZE = N - 2;

    int array[ARRAY_SIZE];

    // Fill array with natural numbers up to N.
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        array[i] = i + 2;
    }
    int num;
    // Set to zero not prime numbers.
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        num = array[i];
        if (num != 0) {
            printf("%i ", num);
            for (int j = num * num - 2; j < ARRAY_SIZE; j += num) {
                array[j] = 0;
            }
        }
    }

    return 0;
}
