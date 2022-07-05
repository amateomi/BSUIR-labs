#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    const int N = 6;
    const int M = 7;
    const int MATRIX_SIZE = N * M;

    int *matrix;

    // Allocate memory for matrix.
    matrix = (int *) malloc(MATRIX_SIZE * sizeof(int));
    if (matrix == NULL) {
        perror("Failed to allocate memory for the matrix!\n");
        return 1;
    }

    // Fill matrix with random numbers and print it.
    srand(time(NULL));
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        *(matrix + i) = rand() % MATRIX_SIZE + 1;

        if (i % M == 0) {
            putchar('\n');
        }
        printf("%3i ", *(matrix + i));
    }

    // Set min value on first even place.
    int minValue = *(matrix + M + 1);
    int iIndex = 1;
    int jIndex = 1;

    // Find suitable number.
    for (int i = 1; i < N; i += 2) {
        for (int j = 1; j < M; j += 2) {
            if (*(matrix + i * M + j) < minValue) {
                minValue = *(matrix + i * M + j);
                iIndex = i;
                jIndex = j;
            }
        }
    }

    printf("\n\nMin value on even places is %i, i = %i, j = %i\n", minValue, iIndex, jIndex);

    return 0;
}
