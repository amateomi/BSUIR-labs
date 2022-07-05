#ifndef LAB1_FUNCTIONS_H
#define LAB1_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

//######################################################################################################################

struct address {
    char *street;
    size_t houseNumber;// 3
    size_t flatNumber;// 3
};
struct customer {
    char *surname;
    char *name;
    char *patronymic;
    struct address address;
    char *phoneNumber;// 7
    char *cardNumber;// 16
};
struct customersInfo {
    struct customer *customers;
    size_t amount;
};

//######################################################################################################################

char menu(void);

char getOption(char leftBorder, char rightBorder);

//######################################################################################################################

struct customersInfo addCustomers(struct customersInfo database);

int getUnsigned(void);

char *getStringLetters(void);

char *getStringDigits(size_t size);

//######################################################################################################################

void showCustomers(struct customersInfo database);

void printSpaces(size_t steps);

//######################################################################################################################

void changeCustomerInformation(struct customersInfo database);

bool changeField(struct customersInfo database, size_t num);

size_t compareWithAmount(size_t amount);

char customerMenu(char *str);

bool continueLoop(char *str);

//######################################################################################################################

struct customersInfo deleteCustomers(struct customersInfo database);

struct customersInfo delete(struct customersInfo database, size_t deleteNum);

//######################################################################################################################

void sortCustomers(struct customersInfo database);

void swap(struct customer *x, struct customer *y);

void sortSurname(struct customersInfo database, bool flag);

void sortName(struct customersInfo database, bool flag);

void sortPatronymic(struct customersInfo database, bool flag);

void sortAddress(struct customersInfo database, bool flag);

void sortPhoneNumber(struct customersInfo database, bool flag);

void sortCardNumber(struct customersInfo database, bool flag);

//######################################################################################################################

void searchCustomers(struct customersInfo database);

char *getKeyWord(size_t *keyWordLen);

bool compare(char *keyWord, size_t keyWordsLen, char *str);

//######################################################################################################################

bool amountZeroCheck(size_t amount);

void freeStruct(struct customer *customers, size_t num);

//######################################################################################################################

#endif //LAB1_FUNCTIONS_H
