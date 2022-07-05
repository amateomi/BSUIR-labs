#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/*----------------------------Declarations-----------------------------------*/

/// Get integer value and check for input errors.
bool getInt(int *number);

/// Recursive check for bracket correctness in the expression.
bool isCorrectBracketExpression(char *string);

/*---------------------------------------------------------------------------*/

int main(void) {
    // Get string size.
    int length;
    do {
        printf("Enter string length:");
    } while (!getInt(&length) || length <= 0);

    // Allocate string memory.
    char *string = (char *) malloc((length + 1) * sizeof(char));
    if (string == NULL) {
        perror("Failed memory allocation!\n");
        exit(1);
    }

    // Get string.
    printf("Enter expression:");
    fgets(string, length + 1, stdin);

    // Print string.
    printf("Your string: %s", string);

    // Check brackets.
    isCorrectBracketExpression(string) ? puts("correct") : puts("not correct");

    free(string);

    return 0;
}

/*###########################################################################*/

bool getInt(int *number) {
    char newLine;

    rewind(stdin);
    if (scanf("%i%c", number, &newLine) != 2 || newLine != '\n') {
        puts("Invalid input! try again.");
        return false;
    }
    return true;
}

/// Return true if ch is '(' or '[' or '{'.
bool isOpenBracket(char ch) {
    return ch == '(' ||
           ch == '[' ||
           ch == '{';
}

/// Return true if ch is ')' or ']' or '}'.
bool isCloseBracket(char ch) {
    return ch == ')' ||
           ch == ']' ||
           ch == '}';
}

bool isCorrectBracketExpression(char *string) {
    static bool isCorrect = true;
    // Store open brackets order.
    static char *stack;
    // Index for stack.
    static int i;

    // Remember open brackets.
    if (isOpenBracket(*string)) {
        // Increase array capacity.
        stack = (char *) realloc(stack, ++i * sizeof(char));
        if (stack == NULL) {
            perror("Failed memory allocation!\n");
            exit(1);
        }
        // Add current open bracket into stack.
        stack[i - 1] = *string;
    } else if (isCloseBracket(*string)) {
        // Stack not allocated and current character is close bracket.
        if (i == 0) {
            isCorrect = false;
            // Check last open bracket.
        } else if (*string == ')') {
            isCorrect = stack[--i] == '(';
        } else if (*string == ']') {
            isCorrect = stack[--i] == '[';
        } else if (*string == '}') {
            isCorrect = stack[--i] == '{';
        }
    }

    // Go out of function if expression is not correct.
    if (isCorrect == false) {
        return isCorrect;
    }

    // Move to the end.
    if (*(string + 1) != '\0') {
        isCorrectBracketExpression(string + 1);
    } else {
        // Current position is last, so stack must be empty.
        isCorrect = i == 0;
        // Free unnecessary memory.
        free(stack);
    }

    return isCorrect;
}

/*###########################################################################*/
