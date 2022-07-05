#ifndef LAB3_FUNCTIONS_H
#define LAB3_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

char menu(void);

char getOption(char leftBorder, char rightBorder);

void addNumbers(char *fileTextName, char *fileBinName);

void showNumbers(char *fileTextName, char *fileBinName);

void findMaxNumber(char *fileName);

void reverseNumberWithIndex(char *fileName);

void countAverageMark(char *fileName);

void swapMaxAndMin(char *fileName);

int getInt(size_t i, const char str[]);

int getUnsigned(void);

char *getFileName(void);

long getNumberOfDigits(int number);

#endif //LAB3_FUNCTIONS_H
