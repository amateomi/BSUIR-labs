#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

/*---------------------------------------------------------------------------*/

/// Get integer value and check for input errors.
bool getInt(int *number);

/// Reallocating string memory.
void reallocateString(char **string, int newSize);

/*---------------------------------------------------------------------------*/

int main() {
    // Get amount of characters for string.
    int length;
    do {
        printf("Enter string length > 0:");
    } while (!getInt(&length) || length <= 0);

    char *string = NULL;
    reallocateString(&string, length + 1);
    // Get string.
    printf("Enter string:");
    fgets(string, length + 1, stdin);
    printf("Your string: %s\n", string);

    int i = 0;
    while (string[i] != '\n') {
        if (string[i] == '*') {
            // Move all characters on 1 position left.
            for (int j = i + 1; j <= length; ++j) {
                string[j - 1] = string[j];
            }
            length--;
            // Reallocate memory.
            reallocateString(&string, length + 1);

        } else {
            length++;
            reallocateString(&string, length + 1);
            // Move all characters on 1 position right.
            for (int j = length; j > i; --j) {
                string[j] = string[j - 1];
            }
            // Step over the forked character.
            i += 2;
        }
    }

    printf("New string: %s\n", string);

    free(string);

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

void reallocateString(char **string, const int newSize) {
    char *temp = NULL;
    temp = (char *) realloc(*string, newSize * sizeof(char));
    if (temp == NULL) {
        perror("Failed memory allocation!\n");
        free(*string);
        exit(1);
    } else {
        *string = temp;
    }
}
/*###########################################################################*/