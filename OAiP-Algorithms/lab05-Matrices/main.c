#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    const int N = 6;

    int matrix[N][N];

    // Fill matrix with random numbers and print it.
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = (rand() % (2 * N)) + 1;
            printf("%3i", matrix[i][j]);
        }
        putchar('\n');
    }

    // Initialize maxValue with first element of matrix.
    int maxValue = matrix[0][0];

    for (int i = 0; i < N / 2 + 1; ++i) {
        for (int j = i; j < N - i; ++j) {
            if (matrix[i][j] > maxValue) {
                maxValue = matrix[i][j];
            }
        }
    }

    printf("\nMax value in triangle is %i\n", maxValue);

    return 0;
}
