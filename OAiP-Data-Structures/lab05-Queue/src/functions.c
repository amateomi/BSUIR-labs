#include "functions.h"
//#####################################################################################################################
bool push(queue_t *queue, const worker_t newElementInfo) {
    node_t *node;
    if (!(node = (node_t *) malloc(sizeof(node_t)))) {
        puts("Not enough memory!");
        system("pause>0");
        return false;
    }
    node->worker = newElementInfo;
    if (!isEmpty(*queue)) {
        queue->tail->next = node;
        queue->tail = node;
    } else
        queue->head = queue->tail = node;
    return true;
}

bool isEmpty(const queue_t queue) {
    return (queue.head) ? false : true;
}

//#####################################################################################################################
char menu(void) {
    char option;
    do {
        system("cls");
        puts("Select an option from the list below:\n");
        puts("1)\t<Add workers>");
        puts("2)\t<Show workers>");
        puts("3)\t<Find workers>");
        puts("4)\t<Delete workers>\n");
        puts("5)\t<Load from file>\n");
        puts("6)\t<Save to file>\n");
        puts("0)\t<Exit the program>");
        putchar('>');
    } while (!(option = getOption('0', '6')));
    return option;
}

//#####################################################################################################################
char *getKeyWordForFind(size_t *keyWordLength) {
    char *str = NULL;
    char c;
    size_t i;
    rewind(stdin);
    for (i = 0; (c = (char) getchar()) != '\n'; i++) {
        // Check
        if (((c < 'a' || c > 'z') && (c < '0' || c > '9') && c != '*') || (i && str[i - 1] == '*' && c == '*')) {
            puts("Invalid input! Try again.");
            system("pause>0");
            return NULL;
        }
        //Reallocation check
        if (!(str = (char *) realloc(str, i + 2))) {
            puts("Not enough memory!");
            system("pause>0");
            return NULL;
        }
        str[i] = c;
    }
    // if only one *
    if (i == 1 && str[0] == '*') {
        puts("Invalid input! Try again.");
        system("pause>0");
        return NULL;
    }
    // Make some magic (edit string for our goals)
    if (str[i - 1] != '*')
        str[i] = '\n';
    else str[i] = '\0';
    for (size_t j = 0; j < i; ++j)
        if (str[j] == '*')
            str[j] = '\0';
    *keyWordLength = i;
    return str;
}

bool compareForFind(char *keyWord, size_t keyWordLength, char *str) {
    char *pos = NULL, *lastPos;
    size_t i = 0, j;
    // if no * on first place
    if (keyWord[0] != '\0')
        for (; keyWord[i] != '\0' && keyWord[i] != '\n'; i++)
            if (keyWord[i] != str[i])
                return false;
    // equality check
    if (keyWord[0] != '\0' && keyWord[keyWordLength] == '\n' && strlen(str) == keyWordLength)
        return true;
    // if no * on last place
    if (keyWord[keyWordLength] == '\n')
        for (keyWordLength--, j = 1; keyWordLength >= 0 && keyWord[keyWordLength] != '\0'; keyWordLength--, j++)
            if (keyWord[keyWordLength] != str[strlen(str) - j])
                return false;
    // check main body
    while (i <= keyWordLength) {
        if (keyWord[i] == '\0') {
            i++;
            continue;
        }
        lastPos = pos;
        if (!(pos = strstr(str, keyWord + i)))
            return false;
        if (pos < lastPos)
            return false;
        i += strlen(keyWord + i);
        str = pos + strlen(keyWord + i);
    }
    return true;
}

void search(queue_t queue) {
    system("cls");
    // Zero check
    if (isEmpty(queue)) {
        puts("Database is empty!");
        system("pause>0");
        return;
    }
    // Get key word for searching
    char *keyWord = NULL;
    size_t keyWordLength = 0;
    do {
        system("cls");
        printf("Enter part of the word to search:\n<example: *ab*c*>\n>");
    } while (!(keyWord = getKeyWordForFind(&keyWordLength)));
    char *convertedNumber = NULL;
    if (!(convertedNumber = (char *) realloc(convertedNumber, 21 * sizeof(char)))) {
        puts("Not enough memory!");
        system("pause>0");
        return;
    }
    //  Check for zero matches
    bool noMatchesCheck = true;
    // Search
    size_t i = 0;
    for (node_t *node = queue.head; node != queue.tail->next; node = node->next, i++) {
        if (compareForFind(keyWord, keyWordLength, _ultoa(node->worker.departmentCode, convertedNumber, 10))) {
            noMatchesCheck = false;
            printf("Match with the %u worker department code:\n%llu\n", i + 1, node->worker.departmentCode);
        }
        if (compareForFind(keyWord, keyWordLength, node->worker.lastName)) {
            noMatchesCheck = false;
            printf("Match with the %u worker lastName:\n%s\n", i + 1, node->worker.lastName);
        }
        if (!node->worker.infoType) {
            if (compareForFind(keyWord, keyWordLength, node->worker.info.date)) {
                noMatchesCheck = false;
                printf("Match with the %u worker worker date:\n", i + 1);
                printDate(node->worker.info.date);
                putchar('\n');
            }
        } else if (compareForFind(keyWord, keyWordLength, _ultoa(node->worker.info.salary, convertedNumber, 10))) {
            noMatchesCheck = false;
            printf("Match with the %u worker salary:\n%llu\n", i + 1, node->worker.info.salary);
        }
    }
    if (noMatchesCheck)
        puts("No matches!");
    system("pause>0");
}

//#####################################################################################################################
char getOption(const char leftBorder, const char rightBorder) {
    char option, newLine;
    // Check
    rewind(stdin);
    if (scanf("%c%c", &option, &newLine) != 2 || newLine != '\n' || option < leftBorder || option > rightBorder) {
        puts("Invalid input, try again!");
        system("pause>0");
        return '\0';
    }
    return option;
}

ull getUnsigned(void) {
    rewind(stdin);
    ull number;
    char c;
    if ((c = (char) getc(stdin)) == '-') {
        puts("Negative numbers cannot be entered! Try again.");
        system("pause>0");
        return 0;
    }
    ungetc(c, stdin);
    if (scanf("%llu%c", &number, &c) != 2 || c != '\n') {
        puts("Invalid input, try again!");
        system("pause>0");
        return 0;
    }
    return number;
}

char *getName(void) {
    char *string = NULL;
    char c;
    size_t i;
    rewind(stdin);
    for (i = 0; (c = (char) getchar()) != '\n'; ++i) {
        // Letter check
        if ((c < 'A' || c > 'Z') && (c < 'a' || c > 'z')) {
            puts("Invalid input, try again!");
            system("pause>0");
            return NULL;
        }
        // Reallocation check
        if (!(string = (char *) realloc(string, i + 2))) {
            puts("Not enough memory!");
            system("pause>0");
            return NULL;
        }
        // Convert letters to the correct case
        (c >= 'A' && c <= 'Z') ? string[i] = (char) (c + ('a' - 'A')) : (string[i] = c);
    }
    if (string) string[i] = '\0';
    return string;
}

bool getBoolPrintMassage(const char *str) {
    char option;
    do {
        system("cls");
        printf("%s", str);
    } while (!(option = getOption('0', '1')));
    return (option == '1') ? true : false;
}

bool getDate(char str[8]) {
    size_t i;
    rewind(stdin);
    for (i = 0; i < 8; i++)
        if (((i == 2 || i == 4) && (char) getchar() != '.') || ((str[i] = (char) getchar()) < '0' || str[i] > '9')) {
            puts("Invalid input, try again!");
            system("pause>0");
            return false;
        }
    if ((str[2] == '1' && str[3] > '2') || str[2] > '1' || str[0] > '3' || (str[0] == '3' && str[1] > 1)) {
        puts("Such date don't exist!");
        system("pause>0");
        return false;
    }
    return true;
}

void printDate(const char *date) {
    for (size_t i = 0; i < 8; ++i) {
        if (i == 2 || i == 4)
            putchar('.');
        putchar(date[i]);
    }
}

size_t getNumberOfDigits(ull number) {
    size_t i;
    for (i = 1; (number /= 10) > 0; ++i);
    return i;
}

void printSpaces(const size_t steps) {
    for (size_t i = 0; i < steps; ++i)
        putchar(' ');
}

char *getFileName(void) {
    char *str = NULL;
    char c;
    size_t i;
    rewind(stdin);
    for (i = 0; (c = (char) getchar()) != '\n'; ++i) {
        // Impossible symbols check
        if (c == '>' || c == '<' || c == ':' || c == '"' || c == '/' || c == '|' || c == '?' || c == '*') {
            puts("forbidden symbols!");
            system("pause>0");
            return NULL;
        }
        //Reallocation check
        if (!(str = (char *) realloc(str, i + 6))) {
            puts("Not enough memory!");
            system("pause>0");
            return NULL;
        }
        str[i] = c;
    }
    if (str) str[i] = '\0';
    // Impossible symbols check
    if (!strcmp(str, "con") || !strcmp(str, "prn") || !strcmp(str, "aux") || !strcmp(str, "nul")) {
        puts("forbidden names!");
        system("pause>0");
        return NULL;
    }
    return str;
}

//#####################################################################################################################
void delete(queue_t *queue) {
    size_t deleteNumber;
    do {
        system("cls");
        // Zero check
        if (isEmpty(*queue)) {
            puts("Database is empty!");
            system("pause>0");
            return;
        }
        do {
            system("cls");
            printf("Enter the number of the user you want to delete:\n>");
        } while (!(deleteNumber = getUnsigned()));
        if (!getBoolPrintMassage("Are you sure?\n1)\t<yes>\n0)\t<no>\n>"))
            continue;
        deleteNodeWithNumber(queue, deleteNumber);
    } while (getBoolPrintMassage("Continue removing customers from the stack?\n1)\t<yes>\n0)\t<no>\n>"));
}

void deleteNodeWithNumber(queue_t *queue, const size_t deleteNumber) {
    size_t i = 1;
    node_t *node, *temp;
    if (deleteNumber == 1) {
        if (queue->head == queue->tail) {
            free(queue->head->worker.lastName);
            free(queue->head);
            queue->head = NULL;
        } else {
            temp = queue->head;
            queue->head = queue->head->next;
            free(temp->worker.lastName);
            free(temp);
        }
        return;
    }
    for (node = queue->head; node != queue->tail; node = node->next, ++i) {
        if ((i + 1) == deleteNumber) {
            if (node->next == queue->tail)
                queue->tail = node;
            temp = node->next;
            node->next = node->next->next;
            free(temp->worker.lastName);
            free(temp);
            return;
        }
    }
    puts("No such number!");
    system("pause>0");
}

//#####################################################################################################################
void add(queue_t *queue) {
    worker_t newWorkerInfo;
    ull numOfNewElements;
    // Get number of elements
    do {
        system("cls");
        printf("How many workers do you want to add?\nEnter the number:\n>");
    } while (!(numOfNewElements = getUnsigned()));
    // Get data for new element
    for (size_t i = 0; i < numOfNewElements; ++i) {
        do {
            system("cls");
            printf("Enter department code for worker %u:\n>", i + 1);
        } while (!(newWorkerInfo.departmentCode = getUnsigned()));
        do {
            system("cls");
            printf("Enter last name for worker %u:\n>", i + 1);
        } while (!(newWorkerInfo.lastName = getString()));
        if ((newWorkerInfo.infoType = getBoolPrintMassage(
                "What will you enter?\n1)\t<salary>\n0)\t<employment date>\n>")))
            do {
                system("cls");
                printf("Enter salary for employee %u:\n>", i + 1);
            } while (!(newWorkerInfo.info.salary = getUnsigned()));
        else
            do {
                system("cls");
                printf("Enter employment date for worker %u:\n<Example: 20.03.2021>\n>", i + 1);
            } while (!(getDate(newWorkerInfo.info.date)));
        if (!push(queue, newWorkerInfo)) return;
    }
}

//#####################################################################################################################
void show(queue_t queue) {
    system("cls");
    // Zero check
    if (isEmpty(queue)) {
        puts("Database is empty!");
        system("pause>0");
        return;
    }
    // Determine the length of the columns
    size_t departmentCodeColumnLen = strlen("Department Code");
    size_t lastNameColumnLen = strlen("Last name");
    size_t dateColumnLen = strlen("Employment date");
    size_t salaryColumnLen = strlen("Salary");
    size_t tempColumnLen;
    node_t *node;
    for (node = queue.head; node != queue.tail->next; node = node->next) {
        if ((tempColumnLen = getNumberOfDigits(node->worker.departmentCode)) > departmentCodeColumnLen)
            departmentCodeColumnLen = tempColumnLen;
        if ((tempColumnLen = strlen(node->worker.lastName)) > lastNameColumnLen)
            lastNameColumnLen = tempColumnLen;
        if (node->worker.infoType == 1 &&
            (tempColumnLen = getNumberOfDigits(node->worker.info.salary)) > salaryColumnLen)
            salaryColumnLen = tempColumnLen;
    }
    size_t steps;
    // Show columns
    system("cls");
    printf("Num| ");
    printSpaces(steps = (departmentCodeColumnLen - strlen("Department Code")) / 2);
    printf("Department Code");
    printSpaces(departmentCodeColumnLen - steps - strlen("Department Code") + 1);
    printf("| ");
    printSpaces(steps = (lastNameColumnLen - strlen("Last name")) / 2);
    printf("Last name");
    printSpaces(lastNameColumnLen - steps - strlen("Last name") + 1);
    printf("| ");
    printSpaces(steps = (dateColumnLen - strlen("Employment date")) / 2);
    printf("Employment date");
    printSpaces(dateColumnLen - steps - strlen("Employment date") + 1);
    printf("| ");
    printSpaces(steps = (salaryColumnLen - strlen("Salary")) / 2);
    printf("Salary");
    printSpaces(salaryColumnLen - steps - strlen("Salary") + 1);
    printf("|\n");
    // Show customers info
    node = queue.head;
    for (size_t i = 0; node != queue.tail->next; ++i, node = node->next) {
        size_t tableLength = departmentCodeColumnLen + lastNameColumnLen + dateColumnLen + salaryColumnLen + 16;
        for (size_t j = 0; j < tableLength; ++j)
            putchar('-');
        putchar('\n');
        printf("%3u", i + 1);
        printf("| ");
        printSpaces(
                steps = (departmentCodeColumnLen - (tempColumnLen = getNumberOfDigits(node->worker.departmentCode))) /
                        2);
        printf("%llu", node->worker.departmentCode);
        printSpaces(departmentCodeColumnLen - steps - tempColumnLen + 1);
        printf("| ");
        printSpaces(steps = (lastNameColumnLen - strlen(node->worker.lastName)) / 2);
        printf("%s", node->worker.lastName);
        printSpaces(lastNameColumnLen - steps - strlen(node->worker.lastName) + 1);
        printf("| ");
        if (node->worker.infoType == 0) {
            printSpaces(steps = (dateColumnLen - 8 - 2) / 2);
            printDate(node->worker.info.date);
            printSpaces(dateColumnLen - steps - 8 - 2 + 1);
        } else {
            printSpaces(steps = (dateColumnLen - strlen("...")) / 2);
            printf("...");
            printSpaces(dateColumnLen - steps - strlen("...") + 1);
        }
        printf("| ");
        if (node->worker.infoType == 1) {
            printSpaces(steps = (salaryColumnLen - (tempColumnLen = getNumberOfDigits(node->worker.info.salary))) / 2);
            printf("%llu", node->worker.info.salary);
            printSpaces(salaryColumnLen - steps - tempColumnLen + 1);
        } else {
            printSpaces(steps = (salaryColumnLen - strlen("...")) / 2);
            printf("...");
            printSpaces(salaryColumnLen - steps - strlen("...") + 1);
        }
        puts("|");
    }
    system("pause>0");
}

//#####################################################################################################################
