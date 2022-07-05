#ifndef LAB7_FUNCTIONS_H
#define LAB7_FUNCTIONS_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#define WORD_LENGTH 7
#define WORD_SIZE (WORD_LENGTH + 1)
#define PRINT_INPUT_ERROR_MESSAGE puts("Invalid input, try again!");
#define PRINT_ALLOCATION_ERROR_MESSAGE puts("Not enough memory!");
#define PRINT_OVERFLOW_MESSAGE printf("You cant input word larger than %i characters!\n", WORD_LENGTH);

// Tree
typedef struct dictionary {
    char englishWord[WORD_SIZE];
    char russianWord[WORD_SIZE];
} dictionary;
typedef struct tree_t {
    dictionary data;
    struct tree_t *left;
    struct tree_t *right;
} tree_t;

tree_t *addNode(tree_t *node, dictionary newWord);

bool emptyCheckTreeAndPrintMessage(tree_t *root);

bool isEmptyRoot(tree_t *root);

// Input
char getOption(char leftBorder, char rightBorder);

bool getEnglishWord(char word[WORD_SIZE]);

void add(tree_t **root);

// Output
char menu(void);

void show(tree_t *node);

void showRecursive(tree_t *node);

void showAsTree(tree_t *node, int space);

// Remove
void clear(tree_t **node);

void deleteWord(tree_t **root);

#endif //LAB7_FUNCTIONS_H
