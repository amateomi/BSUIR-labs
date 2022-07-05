#ifndef LAB4_FUNCTIONS_H
#define LAB4_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

// For simplicity
typedef unsigned long long ull;
union info {
    char date[8];
    ull salary;
};
struct employee {
    ull departmentCode;
    char *lastName;
    bool infoType;
    union info info;
};
struct stack {
    struct employee employee;
    struct stack *next;
};

char menu(void);

char getOption(char leftBorder, char rightBorder);

ull getUnsigned(void);

char *getString(void);

bool getBoolPrintMassage(const char *str);

bool getDate(char str[8]);

void printDate(const char *date);

bool push(struct stack **head, struct employee workerInfo);

void showElements(struct stack *head);

size_t getNumberOfDigits(ull number);

void printSpaces(size_t steps);

char *getFileName(void);

void addElements(struct stack **head);

char *getKeyWordForFind(size_t *keyWordLength);

void findElements(struct stack *head);

bool compareForFind(char *keyWord, size_t keyWordLength, char *str);

void delete(struct stack **head, size_t deleteNumber);

bool isEmpty(const struct stack *head);

void deleteElements(struct stack **head);

void uploadFromFile(struct stack **head);

void saveToTextFile(struct stack *head);

void reverse(struct stack **head);

#endif //LAB4_FUNCTIONS_H
