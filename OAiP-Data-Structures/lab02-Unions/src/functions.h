#ifndef LAB2_FUNCTIONS_H
#define LAB2_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

//######################################################################################################################

typedef bool flag;
// For simplicity
typedef unsigned long long ull;
union information {
    char date[8];
    ull salary;
};
struct worker {
    ull departmentCode;
    char *surname;
    flag infoType;
    union information info;
};
typedef struct workersInfo {
    struct worker *workers;
    size_t amount;
} database;

//######################################################################################################################

char menu(void);

char getOption(char leftBorder, char rightBorder);

//######################################################################################################################

database add(database database);

void show(database database);

void find(database database);

database delete(database database);

//######################################################################################################################

void findMinSalary(database database);

database deleteWorkerWithDate(database database);

//######################################################################################################################

ull getUnsigned(void);

char *getString(void);

bool getDate(char str[8]);

bool getBoolPrintMassage(char *str);

size_t getNumberOfDigits(ull number);

char *getKeyWordForFind(size_t *keyWordLength);

void printSpaces(size_t steps);

void printDate(const char *date);

bool amountZeroCheck(size_t amount);

bool compareForFind(char *keyWord, size_t keyWordLength, char *str);

size_t compareWithAmount(size_t amount);

bool compareDate(const char str1[8], const char str2[8]);

database deleteElement(database database, size_t deleteNum);

//######################################################################################################################

#endif //LAB2_FUNCTIONS_H
