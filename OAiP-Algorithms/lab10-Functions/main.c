#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

/*-----------------------------Declarations----------------------------------*/

/// Insert array as string in matrix on defined position.
void addStringInMatrix(int ***matrix, int amountStrings, int *array, int position);

/// Insert array as column in matrix on defined position.
void addColumnInMatrix(int ***matrix, int amountStrings, int amountColumns, const int *array, int position);

/*+++++++++++++++++++++++++++++Memory control++++++++++++++++++++++++++++++++*/

/// Reallocate and check memory for array.
void reallocateArray(int **array, int size);

/// Allocate and check memory for matrix.
void allocateMatrix(int ***matrix, int amountStrings, int amountColumns);

/// Free matrix memory.
void freeMatrix(int ***matrix, int amountStrings);

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/// Get integer value and check for input errors.
bool getInt(int *number);

/// Fill array with random numbers.
void fillArray(int *array, int size);

/// Fill matrix with random numbers.
void fillMatrix(int **matrix, int amountStrings, int amountColumns);

/// Display array content.
void printArray(int *array, int size);

/// Display matrix content.
void printMatrix(int **matrix, int amountStrings, int amountColumns);

/*---------------------------------------------------------------------------*/

int main() {
    // Start random function
    srand(time(NULL));

    // Get n.
    int n;
    do {
        printf("Enter n:");
    } while (!getInt(&n) || n <= 0);

    // Allocate and initialize matrix.
    int **matrix = NULL;
    allocateMatrix(&matrix, n, n + 1);
    fillMatrix(matrix, n, n + 1);

    // Allocate and initialize arrays.
    int *a = NULL;
    int *b = NULL;
    reallocateArray(&a, n + 1);
    reallocateArray(&b, n + 1);
    fillArray(a, n + 1);
    fillArray(b, n + 1);

    // Show matrix, array a and b.
    puts("Matrix:");
    printMatrix(matrix, n, n + 1);
    puts("Array a:");
    printArray(a, n + 1);
    puts("Array b:");
    printArray(b, n + 1);

    // Get p and q.
    int p, q;
    do {
        printf("Enter p (p<=n):");
    } while (!getInt(&p) || p <= 0 || p > n);
    do {
        printf("Enter q (q<=n+1):");
    } while (!getInt(&q) || q <= 0 || q > n + 1);

    // Insert a.
    addStringInMatrix(&matrix, n, a, p);
    // Insert b.
    addColumnInMatrix(&matrix, n + 1, n + 1, b, q);

    // Show new matrix
    puts("New matrix:");
    printMatrix(matrix, n + 1, n + 2);

    freeMatrix(&matrix, n + 1);
    free(b);

    return 0;
}

/*###########################################################################*/
bool getInt(int *number) {
    char newLine;

    rewind(stdin);
    if (scanf("%i%c", number, &newLine) != 2 || newLine != '\n') {
        puts("Invalid input! try again.\n");
        return false;
    }
    return true;
}

void reallocateArray(int **array, int size) {
    int *temp = NULL;
    temp = (int *) realloc(*array, size * sizeof(int));
    if (temp == NULL) {
        perror("Failed memory allocation in reallocateArray!\n");
        exit(1);
    } else {
        *array = temp;
    }
}

void allocateMatrix(int ***matrix, int amountStrings, int amountColumns) {
    *matrix = (int **) malloc(amountStrings * sizeof(int *));
    if (*matrix == NULL) {
        perror("Failed memory allocation in allocateMatrix!\n");
        exit(1);
    }
    for (int i = 0; i < amountStrings; ++i) {
        (*matrix)[i] = (int *) malloc(amountColumns * sizeof(int));
        if ((*matrix)[i] == NULL) {
            perror("Failed memory allocation in allocateMatrix!\n");
            freeMatrix(matrix, i);
            exit(1);
        }
    }
}

void freeMatrix(int ***matrix, int amountStrings) {
    for (int i = 0; i < amountStrings; ++i) {
        free((*matrix)[i]);
    }
    free(*matrix);
}

void fillArray(int *array, int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = (rand() % size) + 1;
    }
}

void fillMatrix(int **matrix, int amountStrings, int amountColumns) {
    for (int i = 0; i < amountStrings; ++i) {
        for (int j = 0; j < amountColumns; ++j) {
            matrix[i][j] = (rand() % (amountStrings + amountColumns)) + 1;
        }
    }
}

void printArray(int *array, int size) {
    for (int i = 0; i < size; ++i) {
        printf("%5i", array[i]);
    }
    putchar('\n');
}

void printMatrix(int **matrix, int amountStrings, int amountColumns) {
    for (int i = 0; i < amountStrings; ++i) {
        for (int j = 0; j < amountColumns; ++j) {
            printf("%5i", matrix[i][j]);
        }
        putchar('\n');
    }
}

void addStringInMatrix(int ***matrix, int amountStrings, int *array, int position) {
    amountStrings++;
    // Add space for new pointer.
    int **temp = realloc(*matrix, amountStrings * sizeof(int *));
    if (temp == NULL) {
        perror("Failed memory allocation in addStringInMatrix!\n");
        exit(1);
    } else {
        *matrix = temp;
    }

    // Move pointers to add new.
    for (int i = amountStrings - 1; i > position; --i) {
        (*matrix)[i] = (*matrix)[i - 1];
    }

    // Add array pointer in matrix.
    (*matrix)[position] = array;
}

void addColumnInMatrix(int ***matrix, int amountStrings, int amountColumns, const int *array, int position) {
    amountColumns++;
    for (int i = 0; i < amountStrings; ++i) {
        // Add space for new element in each array.
        reallocateArray((*matrix) + i, amountColumns);

        // Move numbers to add new.
        for (int j = amountColumns - 1; j > position; --j) {
            (*matrix)[i][j] = (*matrix)[i][j - 1];
        }

        // Add element for new column.
        (*matrix)[i][position] = array[i];
    }
}
/*###########################################################################*/
