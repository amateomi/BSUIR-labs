#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

///----------------------------------------------------------------------------
// Get integer value and check for input errors.
bool getInt(int *number);

// Allocate memory for matrix.
void allocateMatrix(int ***matrix, int size);

// Free all allocated memory.
void freeMatrix(int ***matrix, int size);

// Fill matrix.
void fillMatrix(int **matrix, int size);

// Print matrix in console.
void printMatrix(const int **matrix, int size);

// Sorting the matrix in the form of a snake.
void snakeSort(int **matrix, int size);

// Swap two integer values.
void swapInt(int *a, int *b);
///----------------------------------------------------------------------------

int main() {
    int matrixSize;
    int **matrix;

    // Get matrix size.
    do {
        printf("Enter matrix size > 0:");
    } while (!getInt(&matrixSize) || matrixSize <= 0);

    allocateMatrix(&matrix, matrixSize);

    printf("Before:");
    fillMatrix(matrix, matrixSize);
    printMatrix((const int **) matrix, matrixSize);

    printf("After:");
    snakeSort(matrix, matrixSize);
    printMatrix((const int **) matrix, matrixSize);

    freeMatrix(&matrix, matrixSize);

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Get integer value and check for input errors.
bool getInt(int *number) {
    char newLine;

    rewind(stdin);
    if (scanf("%i%c", number, &newLine) != 2 || newLine != '\n') {
        puts("Invalid input! try again.\n");
        return false;
    }
    return true;
}

// Allocate memory for matrix.
void allocateMatrix(int ***matrix, const int size) {
    *matrix = (int **) malloc(size * sizeof(int *));
    if (*matrix == NULL) {
        perror("Failed memory allocation!\n");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        *(*matrix + i) = (int *) malloc(size * sizeof(int));
        if (*(*matrix + i) == NULL) {
            freeMatrix(matrix, i + 1);
            perror("Failed memory allocation!\n");
            exit(1);
        }
    }
}

// Free all allocated memory.
void freeMatrix(int ***matrix, const int size) {
    for (int i = 0; i < size; ++i) {
        free(*(*matrix + i));
    }
    free(*matrix);
}

// Fill matrix.
void fillMatrix(int **matrix, const int size) {
    int counter = 1;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = counter++;
        }
    }
}

// Print matrix in console.
void printMatrix(const int **matrix, const int size) {
    putchar('\n');
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%4i", matrix[i][j]);
        }
        putchar('\n');
    }
    putchar('\n');
}

// Sorting the matrix in the form of a snake.
void snakeSort(int **matrix, int size) {
    int maxPos;
    for (int i = 0; i < 2 * size - 1; ++i) {
        maxPos = size - i;

        // Sort top string.
        for (int j = i; j < maxPos; ++j) {
            // Searching in current string.
            for (int k = j + 1; k < maxPos; ++k) {
                if (matrix[i][j] > matrix[i][k]) {
                    swapInt(&matrix[i][k], &matrix[i][j]);
                }
            }
            // Searching in others strings.
            for (int k = i + 1; k < maxPos; ++k) {
                for (int l = i; l < maxPos; ++l) {
                    if (matrix[i][j] > matrix[k][l]) {
                        swapInt(&matrix[i][j], &matrix[k][l]);
                    }
                }
            }
        }

        // Sort right column.
        for (int j = i + 1; j < maxPos; ++j) {
            // Searching in current column.
            for (int k = j + 1; k < maxPos; ++k) {
                if (matrix[j][maxPos - 1] > matrix[k][maxPos - 1]) {
                    swapInt(&matrix[j][maxPos - 1], &matrix[k][maxPos - 1]);
                }
            }
            // Searching in others columns.
            for (int k = i + 1; k < maxPos; ++k) {
                for (int l = i; l < maxPos - 1; ++l) {
                    if (matrix[j][maxPos - 1] > matrix[k][l]) {
                        swapInt(&matrix[j][maxPos - 1], &matrix[k][l]);
                    }
                }
            }
        }

        // Sort bottom string.
        for (int j = maxPos - 1; j >= i; --j) {
            // Searching in current string.
            for (int k = j - 1; k >= i; --k) {
                if (matrix[maxPos - 1][j] > matrix[maxPos - 1][k]) {
                    swapInt(&matrix[maxPos - 1][j], &matrix[maxPos - 1][k]);
                }
            }
            // Searching in others strings.
            for (int k = i + 1; k < maxPos - 1; ++k) {
                for (int l = i; l < maxPos - 1; ++l) {
                    if (matrix[maxPos - 1][j] > matrix[k][l]) {
                        swapInt(&matrix[maxPos - 1][j], &matrix[k][l]);
                    }
                }
            }
        }

        // Sort left column.
        for (int j = maxPos - 1; j > i; --j) {
            // Searching in current column.
            for (int k = j - 1; k > i; --k) {
                if (matrix[j][i] > matrix[k][i]) {
                    swapInt(&matrix[j][i], &matrix[k][i]);
                }
            }
            // Searching in others columns.
            for (int k = i + 1; k < maxPos - 1; ++k) {
                for (int l = i + 1; l < maxPos - 1; ++l) {
                    if (matrix[j][i] > matrix[k][l]) {
                        swapInt(&matrix[j][i], &matrix[k][l]);
                    }
                }
            }
        }

    }
}

// Swap two integer values.
void swapInt(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}
///////////////////////////////////////////////////////////////////////////////
