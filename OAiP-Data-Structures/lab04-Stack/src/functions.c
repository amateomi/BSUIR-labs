#include "functions.h"

char menu(void) {
    char option;
    do {
        system("cls");
        puts("Select an option from the list below:\n");
        puts("1)\t<Add employees>");
        puts("2)\t<Show employees>");
        puts("3)\t<Find employees>");
        puts("4)\t<Delete employees>\n");
        puts("5)\t<Save to files>\n");
        puts("0)\t<Exit the program>");
        putchar('>');
    } while (!(option = getOption('0', '5')));
    return option;
}

char getOption(char leftBorder, char rightBorder) {
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

void addElements(struct stack **head) {
    struct employee element;
    ull numOfNewElements;
    // Get number of elements
    do {
        system("cls");
        printf("How many employee do you want to add?\nEnter the number:\n>");
    } while (!(numOfNewElements = getUnsigned()));
    // Get data for new stack element
    for (size_t i = 0; i < numOfNewElements; ++i) {
        do {
            system("cls");
            printf("Enter department code for employee %u:\n>", i + 1);
        } while (!(element.departmentCode = getUnsigned()));
        do {
            system("cls");
            printf("Enter last name for employee %u:\n>", i + 1);
        } while (!(element.lastName = getString()));
        if ((element.infoType = getBoolPrintMassage("What will you enter?\n1)\t<salary>\n0)\t<employment date>\n>")))
            do {
                system("cls");
                printf("Enter salary for employee %u:\n>", i + 1);
            } while (!(element.info.salary = getUnsigned()));
        else
            do {
                system("cls");
                printf("Enter employment date for employee %u:\n<Example: 20.03.2021>\n>", i + 1);
            } while (!(getDate(element.info.date)));
        if (!push(head, element)) return;
    }
}

bool push(struct stack **head, struct employee workerInfo) {
    struct stack *element;
    // Reallocation check
    if (!(element = malloc(sizeof(struct stack)))) {
        puts("Not enough memory!");
        system("pause>0");
        return false;
    }
    // Add data into ptr
    element->employee = workerInfo;
    // If stack is empty
    if (*head == NULL)
        *head = element;
        // Add element into existing stack
    else {
        element->next = *head;
        *head = element;
    }
    return true;
}

ull getUnsigned(void) {
    ull number;
    char newLine;
    // Check
    rewind(stdin);
    if (scanf("%llu%c", &number, &newLine) != 2 || newLine != '\n') {
        puts("Invalid input, try again!");
        system("pause>0");
        return 0;
    }
    return number;
}

char *getString(void) {
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
        if (((i == 2 || i == 4) && (char) getchar() != '.') || ((str[i] = (char) fgetc(stdin)) < '0' || str[i] > '9')) {
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

void showElements(struct stack *head) {
    // Zero check
    if (isEmpty(head))
        return;
    system("cls");
    // Determine the length of the columns
    size_t departmentCodeColumnLen = strlen("Department Code");
    size_t lastNameColumnLen = strlen("Last name");
    size_t dateColumnLen = strlen("Employment date");
    size_t salaryColumnLen = strlen("Salary");
    size_t tempColumnLen;
    struct stack *element;
    for (element = head; element; element = element->next) {
        if ((tempColumnLen = getNumberOfDigits(element->employee.departmentCode)) > departmentCodeColumnLen)
            departmentCodeColumnLen = tempColumnLen;
        if ((tempColumnLen = strlen(element->employee.lastName)) > lastNameColumnLen)
            lastNameColumnLen = tempColumnLen;
        if (element->employee.infoType == 1 &&
            (tempColumnLen = getNumberOfDigits(element->employee.info.salary)) > salaryColumnLen)
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
    element = head;
    for (size_t i = 0; element; ++i, element = element->next) {
        size_t tableLength = departmentCodeColumnLen + lastNameColumnLen + dateColumnLen + salaryColumnLen + 16;
        for (size_t j = 0; j < tableLength; ++j)
            putchar('-');
        putchar('\n');
        printf("%3u", i + 1);
        printf("| ");
        printSpaces(steps = (departmentCodeColumnLen -
                             (tempColumnLen = getNumberOfDigits(element->employee.departmentCode))) / 2);
        printf("%llu", element->employee.departmentCode);
        printSpaces(departmentCodeColumnLen - steps - tempColumnLen + 1);
        printf("| ");
        printSpaces(steps = (lastNameColumnLen - strlen(element->employee.lastName)) / 2);
        printf("%s", element->employee.lastName);
        printSpaces(lastNameColumnLen - steps - strlen(element->employee.lastName) + 1);
        printf("| ");
        if (element->employee.infoType == 0) {
            printSpaces(steps = (dateColumnLen - 8 - 2) / 2);
            printDate(element->employee.info.date);
            printSpaces(dateColumnLen - steps - 8 - 2 + 1);
        } else {
            printSpaces(steps = (dateColumnLen - strlen("...")) / 2);
            printf("...");
            printSpaces(dateColumnLen - steps - strlen("...") + 1);
        }
        printf("| ");
        if (element->employee.infoType == 1) {
            printSpaces(
                    steps = (salaryColumnLen - (tempColumnLen = getNumberOfDigits(element->employee.info.salary))) / 2);
            printf("%llu", element->employee.info.salary);
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

size_t getNumberOfDigits(ull number) {
    size_t i;
    for (i = 1; (number /= 10) > 0; ++i);
    return i;
}

void printSpaces(size_t steps) {
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

void findElements(struct stack *element) {
    // Zero check
    if (isEmpty(element))
        return;
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
    for (size_t i = 0; element; element = element->next, i++) {
        if (compareForFind(keyWord, keyWordLength, _ultoa(element->employee.departmentCode, convertedNumber, 10))) {
            noMatchesCheck = false;
            printf("Match with the %u employee department code:\n%llu\n", i + 1, element->employee.departmentCode);
        }
        if (compareForFind(keyWord, keyWordLength, element->employee.lastName)) {
            noMatchesCheck = false;
            printf("Match with the %u employee lastName:\n%s\n", i + 1, element->employee.lastName);
        }
        if (!element->employee.infoType) {
            if (compareForFind(keyWord, keyWordLength, element->employee.info.date)) {
                noMatchesCheck = false;
                printf("Match with the %u employee employment date:\n", i + 1);
                printDate(element->employee.info.date);
                putchar('\n');
            }
        } else if (compareForFind(keyWord, keyWordLength, _ultoa(element->employee.info.salary, convertedNumber, 10))) {
            noMatchesCheck = false;
            printf("Match with the %u employee salary:\n%llu\n", i + 1, element->employee.info.salary);
        }
    }
    if (noMatchesCheck)
        puts("No matches!");
    system("pause>0");
}

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

void deleteElements(struct stack **head) {
    size_t deleteNumber;
    do {
        // Zero check
        if (isEmpty(*head))
            return;
        system("cls");
        do {
            system("cls");
            printf("Enter the number of the user you want to remove:\n>");
        } while (!(deleteNumber = getUnsigned()));
        if (!getBoolPrintMassage("Are you sure?\n1)\t<yes>\n0)\t<no>\n>"))
            continue;
        delete(head, deleteNumber);
    } while (getBoolPrintMassage("Continue removing customers from the stack?\n1)\t<yes>\n0)\t<no>\n>"));
}

void delete(struct stack **head, size_t deleteNumber) {
    struct stack *temp, *element = *head;
    // Delete first element
    if (deleteNumber == 1) {
        *head = element->next;
        free(element->employee.lastName);
        free(element);
        return;
    }
    // Delete next element in loop
    for (size_t i = 2; element; element = element->next, i++)
        if (i == deleteNumber) {
            temp = element->next;
            element->next = element->next->next;
            free(temp->employee.lastName);
            free(temp);
            return;
        }
    // Delete number correct check
    system("cls");
    puts("No such employee number!");
    system("pause>0");
}

bool isEmpty(const struct stack *head) {
    if (head == NULL) {
        puts("Database is empty!");
        system("pause>0");
        return true;
    }
    return false;
}