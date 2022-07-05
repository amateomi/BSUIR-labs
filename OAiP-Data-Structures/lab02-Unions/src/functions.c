#include "functions.h"

//######################################################################################################################
char menu(void) {
    char option;
    do {
        system("cls");
        puts("Select an option from the list below:\n");
        puts("1)\t<Add workers>");
        puts("2)\t<Show workers>");
        puts("3)\t<Find workers>");
        puts("4)\t<Delete workers>\n");
        puts("Additional tasks:\n");
        puts("5)\t<Find workers with a minimum salary>");
        puts("6)\t<Delete workers on a given hiring day>\n");
        puts("0)\t<Exit the program>");
        putchar('>');
    } while (!(option = getOption('0', '6')));
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

//######################################################################################################################
database add(database database) {
    // Get number of new workers
    ull number;
    do {
        system("cls");
        puts("How many customers do you want to add? Enter the number:");
        putchar('>');
    } while (!(number = getUnsigned()));
    // Reallocation check
    struct worker *temp;
    database.amount += number;
    if (!(temp = (struct worker *) realloc(database.workers, database.amount * sizeof(struct worker)))) {
        database.amount -= number;
        puts("Not enough memory to add new workers!");
        system("pause>0");
        return database;
    }
    database.workers = temp;
    // Add information
    for (size_t i = database.amount - number; i < database.amount; ++i) {
        do {
            system("cls");
            printf("Enter department code for worker %u:\n>", i + 1);
        } while (!(database.workers[i].departmentCode = getUnsigned()));
        do {
            system("cls");
            printf("Enter surname for worker %u:\n>", i + 1);
        } while (!(database.workers[i].surname = getString()));
        if ((database.workers[i].infoType = getBoolPrintMassage(
                "What will you enter?\n0)\t<employment date>\n1)\t<salary>\n>")))
            do {
                system("cls");
                printf("Enter salary for worker %u:\n>", i + 1);
            } while (!(database.workers[i].info.salary = getUnsigned()));
        else
            do {
                system("cls");
                printf("Enter employment date for worker %u:\n<Example: 20.03.2021>\n>", i + 1);
            } while (!(getDate(database.workers[i].info.date)));
    }
    return database;
}

void show(database database) {
    // Zero check
    if (amountZeroCheck(database.amount))
        return;
    // Determine the length of the columns
    size_t departmentCodeColumnLength = strlen("Department Code");
    size_t surnameColumnLength = strlen("Surname");
    size_t dateColumnLength = strlen("Employment date");
    size_t salaryColumnLength = strlen("Salary");
    size_t buffer;
    for (size_t i = 0; i < database.amount; ++i) {
        if ((buffer = getNumberOfDigits(database.workers[i].departmentCode)) > departmentCodeColumnLength)
            departmentCodeColumnLength = buffer;
        if (strlen(database.workers[i].surname) > surnameColumnLength)
            surnameColumnLength = strlen(database.workers[i].surname);
        if (database.workers[i].infoType == 1 &&
            (buffer = getNumberOfDigits(database.workers[i].info.salary)) > salaryColumnLength)
            salaryColumnLength = buffer;
    }
    size_t steps;
    // Show columns
    system("cls");
    printf("Num| ");
    printSpaces(steps = (departmentCodeColumnLength - strlen("Department Code")) / 2);
    printf("Department Code");
    printSpaces(departmentCodeColumnLength - steps - strlen("Department Code") + 1);
    printf("| ");
    printSpaces(steps = (surnameColumnLength - strlen("Surname")) / 2);
    printf("Surname");
    printSpaces(surnameColumnLength - steps - strlen("Surname") + 1);
    printf("| ");
    printSpaces(steps = (dateColumnLength - strlen("Employment date")) / 2);
    printf("Employment date");
    printSpaces(dateColumnLength - steps - strlen("Employment date") + 1);
    printf("| ");
    printSpaces(steps = (salaryColumnLength - strlen("Salary")) / 2);
    printf("Salary");
    printSpaces(salaryColumnLength - steps - strlen("Salary") + 1);
    printf("|\n");
    // Show customers information
    for (size_t i = 0; i < database.amount; ++i) {
        size_t tableLength =
                departmentCodeColumnLength + surnameColumnLength + dateColumnLength + salaryColumnLength + 16;
        for (size_t j = 0; j < tableLength; ++j)
            putchar('-');
        putchar('\n');

        printf("%3u", i + 1);

        printf("| ");
        printSpaces(steps = (departmentCodeColumnLength -
                             (buffer = getNumberOfDigits(database.workers[i].departmentCode))) / 2);
        printf("%llu", database.workers[i].departmentCode);
        printSpaces(departmentCodeColumnLength - steps - buffer + 1);

        printf("| ");
        printSpaces(steps = (surnameColumnLength - strlen(database.workers[i].surname)) / 2);
        printf("%s", database.workers[i].surname);
        printSpaces(surnameColumnLength - steps - strlen(database.workers[i].surname) + 1);

        printf("| ");
        if (database.workers[i].infoType == 0) {
            printSpaces(steps = (dateColumnLength - 8 - 2) / 2);
            printDate(database.workers[i].info.date);
            printSpaces(dateColumnLength - steps - 8 - 2 + 1);
        } else {
            printSpaces(steps = (dateColumnLength - strlen("...")) / 2);
            printf("...");
            printSpaces(dateColumnLength - steps - strlen("...") + 1);
        }

        printf("| ");
        if (database.workers[i].infoType == 1) {
            printSpaces(
                    steps = (salaryColumnLength - (buffer = getNumberOfDigits(database.workers[i].info.salary))) / 2);
            printf("%llu", database.workers[i].info.salary);
            printSpaces(salaryColumnLength - steps - buffer + 1);
        } else {
            printSpaces(steps = (salaryColumnLength - strlen("...")) / 2);
            printf("...");
            printSpaces(salaryColumnLength - steps - strlen("...") + 1);
        }

        puts("|");
    }
    system("pause>0");
}

void find(database database) {
    // Zero check
    if (amountZeroCheck(database.amount))
        return;
    // Get key word for searching
    char *keyWord = NULL;
    size_t keyWordLength = 0;
    do {
        system("cls");
        puts("Enter part of the word to search:");
        puts("<example: *ab*c*>");
        putchar('>');
    } while (!(keyWord = getKeyWordForFind(&keyWordLength)));
    // Need for _ultoa
    char *temp = NULL;
    if (!(temp = (char *) realloc(temp, 21 * sizeof(char)))) {
        puts("Not enough memory!");
        system("pause>0");
        return;
    }
    // Flag is checking for zero matches
    flag noMatchesCheck = true;
    for (size_t i = 0; i < database.amount; ++i) {
        if (compareForFind(keyWord, keyWordLength, _ultoa(database.workers[i].departmentCode, temp, 10))) {
            noMatchesCheck = false;
            printf("Match with the %u worker department code\t%llu\n", i + 1, database.workers[i].departmentCode);
        }
        if (compareForFind(keyWord, keyWordLength, database.workers[i].surname)) {
            noMatchesCheck = false;
            printf("Match with the %u worker surname\t%s\n", i + 1, database.workers[i].surname);
        }
        if (!database.workers[i].infoType) {
            if (compareForFind(keyWord, keyWordLength, database.workers[i].info.date)) {
                noMatchesCheck = false;
                printf("Match with the %u worker employment date\t", i + 1);
                printDate(database.workers[i].info.date);
                putchar('\n');
            }
        } else {
            if (compareForFind(keyWord, keyWordLength, _ultoa(database.workers[i].info.salary, temp, 10))) {
                noMatchesCheck = false;
                printf("Match with the %u worker salary\t%llu\n", i + 1, database.workers[i].info.salary);
            }
        }
    }
    if (noMatchesCheck)
        puts("No matches!");
    system("pause>0");
}

database delete(database database) {
    size_t posDelete;
    do {
        // Zero check
        if (amountZeroCheck(database.amount))
            return database;
        system("cls");
        do
            printf("Enter the number of the user you want to remove:\n>");
        while (!(posDelete = compareWithAmount(database.amount)));
        if (!getBoolPrintMassage("Are you sure?\n<1) - yes, 0) - no>\n>"))
            continue;
        database = deleteElement(database, posDelete - 1);
    } while (getBoolPrintMassage("continue removing customers from the database?\n<1) - yes, 0) - no>\n>"));
    return database;
}

//######################################################################################################################
void findMinSalary(database database) {
    // Zero check
    if (amountZeroCheck(database.amount))
        return;
    system("cls");
    ull salaryMin;
    flag isNoSalary = true;
    // Find first salary
    for (size_t i = 0; i < database.amount; ++i)
        if (database.workers[i].infoType) {
            salaryMin = database.workers[i].info.salary;
            isNoSalary = false;
        }
    // If no salary in the database
    if (isNoSalary) {
        puts("There is no workers with salary!");
        system("pause>0");
        return;
    }
    // Find min salary
    for (size_t i = 0; i < database.amount; ++i)
        if (database.workers[i].infoType && database.workers[i].info.salary < salaryMin)
            salaryMin = database.workers[i].info.salary;
    printf("Minimal salary is %llu\n", salaryMin);
    // Show numbers of the cheapest workers
    puts("Number of workers with minimal salary:");
    for (size_t i = 0; i < database.amount; ++i)
        if (database.workers[i].infoType && database.workers[i].info.salary == salaryMin)
            printf("%u ", i + 1);
    system("pause>0");
}

database deleteWorkerWithDate(database database) {
    // Zero check
    if (amountZeroCheck(database.amount))
        return database;
    char deleteDate[8];
    // Get hiring date
    do {
        system("cls");
        printf("Enter hiring date for workers:\n<Example: 20.03.2021>\n>");
    } while (!(getDate(deleteDate)));
    if (!getBoolPrintMassage("Are you sure?\n<1) - yes, 0) - no>\n>"))
        return database;
    // Delete
    flag isNoMatches = true;
    for (size_t i = 0; i < database.amount; ++i)
        if (!database.workers[i].infoType && compareDate(database.workers[i].info.date, deleteDate)) {
            database = deleteElement(database, i);
            isNoMatches = false;
        }
    if (isNoMatches) {
        puts("There is no such date!");
        system("pause>0");
    }
    return database;
}

//######################################################################################################################
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
    char *str = NULL;
    char c;
    size_t i;
    rewind(stdin);
    for (i = 0; (c = (char) getchar()) != '\n'; ++i) {
        //Letter check
        if ((c < 'A' || c > 'Z') && (c < 'a' || c > 'z')) {
            puts("Invalid input, try again!");
            system("pause>0");
            return NULL;
        }
        //Reallocation check
        if (!(str = (char *) realloc(str, i + 2))) {
            puts("Not enough memory!");
            system("pause>0");
            return NULL;
        }
        // Convert letters to the correct case
        (c >= 'A' && c <= 'Z') ? str[i] = (char) (c + ('a' - 'A')) : (str[i] = c);
    }
    if (str) str[i] = '\0';
    return str;
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

bool getBoolPrintMassage(char *str) {
    char option;
    do {
        system("cls");
        printf("%s", str);
    } while (!(option = getOption('0', '1')));
    return (option == '1') ? true : false;
}

size_t getNumberOfDigits(ull number) {
    size_t i;
    for (i = 1; (number /= 10) > 0; ++i);
    return i;
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

void printSpaces(size_t steps) {
    for (size_t i = 0; i < steps; ++i)
        putchar(' ');
}

void printDate(const char *date) {
    for (size_t j = 0; j < 8; ++j) {
        if (j == 2 || j == 4)
            putchar('.');
        putchar(date[j]);
    }
}

bool amountZeroCheck(size_t amount) {
    if (amount == 0) {
        puts("There is no customer information in the database!");
        system("pause>0");
        return true;
    }
    return false;
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

size_t compareWithAmount(size_t amount) {
    size_t num;
    if (amount < (num = getUnsigned()) || num == 0) {
        puts("There is no such customer! Try again.");
        system("pause>0");
        system("cls");
        return 0;
    }
    return num;
}

bool compareDate(const char str1[8], const char str2[8]) {
    for (size_t i = 0; i < 8; ++i)
        if (str1[i] != str2[i])
            return false;
    return true;
}

database deleteElement(database database, size_t deleteNum) {
    // Remember last struct
    struct worker buffer = database.workers[database.amount - 1];
    // If there is only 1 struct
    if (database.amount == 1) {
        database.amount = 0;
        free(database.workers[0].surname);
        free(database.workers);
        return database;
    }
    // Reallocation check
    struct worker *temp;
    if (!(temp = (struct worker *) realloc(database.workers, (database.amount - 1) * sizeof(struct worker)))) {
        puts("Not enough memory!");
        system("pause>0");
        return database;
    }
    database.workers = temp;
    database.amount--;
    // If we're deleting last struct
    if (database.amount == deleteNum)
        return database;
    // Transfer information from struct to struct
    size_t i;
    for (i = deleteNum; i < database.amount - 1; ++i)
        database.workers[i] = database.workers[i + 1];
    database.workers[i] = buffer;
    return database;
}
//######################################################################################################################