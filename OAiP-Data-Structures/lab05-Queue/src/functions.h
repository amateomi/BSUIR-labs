#ifndef LAB5_FUNCTIONS_H
#define LAB5_FUNCTIONS_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef unsigned long long ull;
typedef union info {
    char date[8];
    ull salary;
} info_t;
typedef struct worker {
    ull departmentCode;
    char *lastName;
    bool infoType;
    info_t info;
} worker_t;
typedef struct node_t {
    worker_t worker;
    struct node_t *next;
} node_t;
typedef struct queue {
    node_t *head, *tail;
} queue_t;

// Queue
bool push(queue_t *queue, worker_t newElementInfo);

bool isEmpty(queue_t queue);

// Menu
char menu(void);

// get-print
ull getUnsigned(void);

char getOption(char leftBorder, char rightBorder);

char *getName(void);

bool getBoolPrintMassage(const char *str);

bool getDate(char str[8]);

void printDate(const char *date);

size_t getNumberOfDigits(ull number);

void printSpaces(size_t steps);

char *getFileName(void);


void add(queue_t *queue);

void show(queue_t queue);

void search(queue_t queue);

void delete(queue_t *queue);

void deleteNodeWithNumber(queue_t *queue, size_t deleteNumber);

// File
void load(queue_t *queue);

void save(queue_t queue);

#endif //LAB5_FUNCTIONS_H
