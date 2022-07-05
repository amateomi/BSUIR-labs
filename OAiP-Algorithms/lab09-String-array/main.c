#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

/*---------------------------------------------------------------------------*/
/// Get integer value and check for input errors.
bool getInt(int *number);

/// Get string according to task.
void getText(char **text, int amountString, int stringLength);

/// Get single string and check for correctness.
bool getString(char *string, int stringLength);

/// Print text.
void printText(const char **text, int amountString);

bool isLowLetter(char ch);

bool isDigit(char ch);

bool isArithmeticOperation(char ch);

/// Count characters groups and print result.
void analyzeText(char **text, int amountStrings);

/// Skip whole word in string.
void skipWord(char *string, int *j, bool type(const char));

/// Free text memory.
void freeText(char ***text, int amountStrings);

/*---------------------------------------------------------------------------*/

int main() {
    // Get text size.
    int amountStrings;
    do {
        printf("Enter amount of strings in text:");
    } while (!getInt(&amountStrings) || amountStrings <= 0);
    int stringLength;
    do {
        printf("Enter string length:");
    } while (!getInt(&stringLength) || stringLength <= 0);

    // Allocating memory for text.
    char **text = (char **) malloc(amountStrings * sizeof(char *));
    if (text == NULL) {
        perror("Failed memory allocation!\n");
        exit(1);
    }
    for (int i = 0; i < amountStrings; ++i) {
        text[i] = (char *) malloc((stringLength + 1) * sizeof(char));
        if (text[i] == NULL) {
            perror("Failed memory allocation!\n");
            freeText(&text, i);
            exit(1);
        }
    }

    getText(text, amountStrings, stringLength);
    printText((const char **) text, amountStrings);

    analyzeText(text, amountStrings);

    freeText(&text, amountStrings);

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

void getText(char **text, const int amountString, const int stringLength) {
    for (int i = 0; i < amountString; ++i) {
        printf("Enter string No %i:", i + 1);
        rewind(stdin);
        if (!getString(text[i], stringLength)) {
            i--;
        }
    }
}

bool getString(char *string, const int stringLength) {
    int i;
    char ch;

    for (i = 0; i < stringLength; ++i) {
        ch = (char) getchar();
        if (ch == '\n') {
            break;
        } else if (isLowLetter(ch) || isDigit(ch) || isArithmeticOperation(ch)) {
            string[i] = ch;
        } else {
            puts("Invalid input! Try again.");
            return false;
        }
    }

    string[i] = '\0';
    return true;
}

void printText(const char **text, const int amountString) {
    for (int i = 0; i < amountString; ++i) {
        puts(text[i]);
    }
}

inline bool isLowLetter(const char ch) {
    return ch >= 'a' && ch <= 'z';
}

inline bool isDigit(const char ch) {
    return ch >= '0' && ch <= '9';
}

inline bool isArithmeticOperation(const char ch) {
    return ch == '+' || ch == '-' || ch == '*';
}

void analyzeText(char **text, const int amountStrings) {
    int amountLetterWords = 0;
    int amountDigitsWords = 0;
    int amountArithmeticOperationWords = 0;

    char ch;

    for (int i = 0; i < amountStrings; ++i) {
        int j = 0;
        // I use pointer to function, because it's simplify code structure.
        while ((ch = text[i][j]) != '\0') {
            if (isLowLetter(ch)) {
                amountLetterWords++;
                skipWord(text[i], &j, isLowLetter);
            } else if (isDigit(ch)) {
                amountDigitsWords++;
                skipWord(text[i], &j, isDigit);
            } else if (isArithmeticOperation(ch)) {
                amountArithmeticOperationWords++;
                skipWord(text[i], &j, isArithmeticOperation);
            }
        }
    }

    printf("In text %i letter groups, %i digits groups, %i arithmetic groups.\n",
           amountLetterWords, amountDigitsWords, amountArithmeticOperationWords);
}

void skipWord(char *string, int *j, bool (*type)(char)) {
    while (type(string[++(*j)]));
}

void freeText(char ***text, const int amountStrings) {
    for (int i = 0; i < amountStrings; ++i) {
        free((*text)[i]);
    }
    free(*text);
}

/*###########################################################################*/