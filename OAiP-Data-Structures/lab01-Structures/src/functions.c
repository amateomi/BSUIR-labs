#include "functions.h"

//######################################################################################################################
//Menu settings
//######################################################################################################################
char menu(void) {
    char option;
    do {
        puts("select an option from the list below:");
        puts("1)\t<Add customers>");
        puts("2)\t<Show customers>");
        puts("3)\t<Change customers>");
        puts("4)\t<Find customers>");
        puts("5)\t<Sort customers>");
        puts("6)\t<Remove customers>");
        puts("0)\t<Exit the program>");
        printf(">");
    } while (!(option = getOption('0', '6')));
    return option;
}

char getOption(char leftBorder, char rightBorder) {
    char option, newLine;
    rewind(stdin);
    // Check
    if (scanf("%c%c", &option, &newLine) != 2 || newLine != '\n' || option < leftBorder || option > rightBorder) {
        puts("Invalid input, try again!");
        system("pause>0");
        system("cls");
        return '\0';
    }
    return option;
}

//######################################################################################################################
//Add customer information
//######################################################################################################################
struct customersInfo addCustomers(struct customersInfo database) {
    // Get number of new customers
    int number;
    do {
        system("cls");
        puts("How many customers do you want to add? Enter the number:");
        puts("<0 - back to menu>");
        putchar('>');
    } while ((number = getUnsigned()) < 0);
    if (number == 0)
        return database;
    // Reallocation check
    struct customer *temp = NULL;
    database.amount += number;
    if (!(temp = (struct customer *) realloc(database.customers, database.amount * sizeof(struct customer)))) {
        database.amount -= number;
        puts("Not enough memory to add new customers!");
        system("pause>0");
        return database;
    }
    database.customers = temp;

    // Add information
    for (size_t i = database.amount - number; i < database.amount; ++i) {
        do {
            system("cls");
            printf("Enter surname for customer %u:\n>", i + 1);
        } while (!(database.customers[i].surname = getStringLetters()));
        do {
            system("cls");
            printf("Enter name for customer %u:\n>", i + 1);
        } while (!(database.customers[i].name = getStringLetters()));
        do {
            system("cls");
            printf("Enter patronymic for customer %u:\n>", i + 1);
        } while (!(database.customers[i].patronymic = getStringLetters()));
        do {
            system("cls");
            printf("Enter street for customer %u:\n>", i + 1);
        } while (!(database.customers[i].address.street = getStringLetters()));
        do {
            system("cls");
            printf("Enter house number for customer %u:\n<3 char max>\n>", i + 1);
        } while (!(database.customers[i].address.houseNumber = getUnsigned()) ||
                 database.customers[i].address.houseNumber > 999);
        do {
            system("cls");
            printf("Enter flat number for customer %u:\n<3 char max>\n>", i + 1);
        } while (!(database.customers[i].address.flatNumber = getUnsigned()) ||
                 database.customers[i].address.flatNumber > 999);
        do {
            system("cls");
            printf("Enter phone number for customer %u:\n", i + 1);
            printf("<Phone number must have 7 digits>\n>");
        } while (!(database.customers[i].phoneNumber = getStringDigits(7)));
        do {
            system("cls");
            printf("Enter card number for customer %u:\n", i + 1);
            printf("<Card number must have 16 digits>\n>");
        } while (!(database.customers[i].cardNumber = getStringDigits(16)));
    }
    return database;
}

int getUnsigned(void) {
    int num;
    char newLine;
    rewind(stdin);
    // Check
    if (scanf("%i%c", &num, &newLine) != 2 || newLine != '\n' || num < 0) {
        puts("Invalid input, try again!");
        system("pause>0");
        return -1;
    }
    return num;
}

char *getStringLetters(void) {
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

char *getStringDigits(size_t size) {
    char *str = NULL;
    // Allocation check
    if (!(str = (char *) malloc(size + 1))) {
        puts("Not enough memory!");
        system("pause>0");
        return NULL;
    }
    rewind(stdin);
    for (size_t i = 0; i < size; ++i) {
        str[i] = (char) getchar();
        if (str[i] < '0' || str[i] > '9') {
            puts("Invalid input, try again!");
            system("pause>0");
            return NULL;
        }
    }
    if (getchar() != '\n') {
        puts("Too many character!");
        system("pause>0");
        return NULL;
    }
    str[size] = '\0';
    return str;
}

//######################################################################################################################
//Show customer information
//######################################################################################################################
void showCustomers(struct customersInfo database) {
    // Zero check
    if (amountZeroCheck(database.amount))
        return;
    // Determine the length of the columns
    size_t surnameColumnLen = strlen("Surname");
    size_t nameColumnLen = strlen("Name");
    size_t patronymicColumnLen = strlen("Patronymic");
    size_t addressColumnLen = 1;
    size_t phoneNumberLen = strlen("Phone Number");
    size_t cardNumberLen = 19;
    for (size_t i = 0; i < database.amount; ++i) {
        if (strlen(database.customers[i].surname) > surnameColumnLen)
            surnameColumnLen = strlen(database.customers[i].surname);
        if (strlen(database.customers[i].name) > nameColumnLen)
            nameColumnLen = strlen(database.customers[i].name);
        if (strlen(database.customers[i].patronymic) > patronymicColumnLen)
            patronymicColumnLen = strlen(database.customers[i].patronymic);
        if (strlen(database.customers[i].address.street) > addressColumnLen)
            addressColumnLen = strlen(database.customers[i].address.street);
    }
    addressColumnLen += 13;

    size_t steps;
    // Show columns
    system("cls");
    printf("%3s| ", "Num");
    printSpaces(steps = (surnameColumnLen - strlen("Surname")) / 2);
    printf("Surname");
    printSpaces(surnameColumnLen - steps - strlen("Surname") + 1);

    printf("| ");
    printSpaces(steps = (nameColumnLen - strlen("Name")) / 2);
    printf("Name");
    printSpaces(nameColumnLen - steps - strlen("Name") + 1);

    printf("| ");
    printSpaces(steps = (patronymicColumnLen - strlen("Patronymic")) / 2);
    printf("Patronymic");
    printSpaces(patronymicColumnLen - steps - strlen("Patronymic") + 1);

    printf("| ");
    printSpaces(steps = (addressColumnLen - strlen("Address")) / 2);
    printf("Address");
    printSpaces(addressColumnLen - steps - strlen("Address") + 1);

    printf("| ");
    printSpaces(steps = (phoneNumberLen - strlen("Phone Number")) / 2);
    printf("Phone Number");
    printSpaces(phoneNumberLen - steps - strlen("Phone Number") + 1);

    printf("| ");
    printSpaces(steps = (cardNumberLen - strlen("Card Number")) / 2);
    printf("Card Number");
    printSpaces(cardNumberLen - steps - strlen("Card Number") + 1);

    printf("|\n");

    // Show customers information
    for (size_t i = 0; i < database.amount; ++i) {
        size_t max = surnameColumnLen + nameColumnLen + patronymicColumnLen + addressColumnLen + phoneNumberLen +
                     cardNumberLen + 22;
        for (size_t j = 0; j < max; ++j)
            putchar('-');
        puts("");

        printf("%3u", i + 1);

        printf("| ");
        printSpaces(steps = (surnameColumnLen - strlen(database.customers[i].surname)) / 2);
        printf("%s", database.customers[i].surname);
        printSpaces(surnameColumnLen - steps - strlen(database.customers[i].surname) + 1);

        printf("| ");
        printSpaces(steps = (nameColumnLen - strlen(database.customers[i].name)) / 2);
        printf("%s", database.customers[i].name);
        printSpaces(nameColumnLen - steps - strlen(database.customers[i].name) + 1);

        printf("| ");
        printSpaces(steps = (patronymicColumnLen - strlen(database.customers[i].patronymic)) / 2);
        printf("%s", database.customers[i].patronymic);
        printSpaces(patronymicColumnLen - steps - strlen(database.customers[i].patronymic) + 1);

        printf("| ");
        printSpaces(steps = (addressColumnLen - (strlen(database.customers[i].address.street) + 13)) / 2);
        printf("%s ", database.customers[i].address.street);
        printf("h.%3u ", database.customers[i].address.houseNumber);
        printf("fl.%3u", database.customers[i].address.flatNumber);
        printSpaces(addressColumnLen - steps - (strlen(database.customers[i].address.street) + 13) + 1);

        printf("| ");
        printSpaces(steps = (phoneNumberLen - (strlen(database.customers[i].phoneNumber) + 2)) / 2);
        for (size_t j = 0; database.customers[i].phoneNumber[j] != '\0'; ++j) {
            if (j == 3 || j == 5)
                putchar('-');
            putchar(database.customers[i].phoneNumber[j]);
        }
        printSpaces(phoneNumberLen - steps - (strlen(database.customers[i].phoneNumber) + 2) + 1);

        printf("| ");
        printSpaces(steps = (cardNumberLen - (strlen(database.customers[i].cardNumber) + 3)) / 2);
        for (size_t j = 0; database.customers[i].cardNumber[j] != '\0'; ++j) {
            if (j == 4 || j == 8 || j == 12)
                putchar(' ');
            putchar(database.customers[i].cardNumber[j]);
        }
        printSpaces(cardNumberLen - steps - (strlen(database.customers[i].cardNumber) + 3) + 1);

        puts("|");
    }
    system("pause>0");
}

void printSpaces(size_t steps) {
    for (size_t i = 0; i < steps; ++i)
        printf(" ");
}

//######################################################################################################################
// Change customer information
//######################################################################################################################
void changeCustomerInformation(struct customersInfo database) {
    // Zero check
    if (amountZeroCheck(database.amount))
        return;

    size_t num;
    do {
        system("cls");
        do
            printf("Enter the number of the user you want to change:\n>");
        while (!(num = compareWithAmount(database.amount)));
        do
            if (!changeField(database, num - 1)) break;
        while (continueLoop("continue to modify this user's information?\n<1) - yes, 0) - no>"));
    } while (continueLoop("continue to make changes to the database?\n<1) - yes, 0) - no>"));
}

bool changeField(struct customersInfo database, size_t num) {
    char *str;
    size_t house, flat;
    switch (customerMenu("select the field to change from the list below:")) {
        case '1':
            system("cls");
            printf("Enter new surname for customer %u:\n>", num + 1);
            while (!(str = getStringLetters()));
            if (continueLoop("Are you sure?\n<1) - yes, 0) - no>"))
                database.customers[num].surname = str;
            break;
        case '2':
            system("cls");
            printf("Enter new name for customer %u:\n>", num + 1);
            while (!(str = getStringLetters()));
            if (continueLoop("Are you sure?\n<1) - yes, 0) - no>"))
                database.customers[num].name = str;
            break;
        case '3':
            system("cls");
            printf("Enter new patronymic for customer %u:\n>", num + 1);
            while (!(str = getStringLetters()));
            if (continueLoop("Are you sure?\n<1) - yes, 0) - no>"))
                database.customers[num].patronymic = str;
            break;
        case '4':
            system("cls");
            printf("Enter new street for customer %u:\n>", num + 1);
            while (!(str = getStringLetters()));
            printf("Enter new house number for customer %u:\n>", num + 1);
            while (!(house = getUnsigned()) || house > 999);
            printf("Enter new flat number for customer %u:\n", num + 1);
            while (!(flat = getUnsigned()) || flat > 999);
            if (continueLoop("Are you sure?\n<1) - yes, 0) - no>")) {
                database.customers[num].address.street = str;
                database.customers[num].address.houseNumber = house;
                database.customers[num].address.flatNumber = flat;
            }
            break;
        case '5':
            system("cls");
            printf("Enter new phone number for customer %u:\n>", num + 1);
            while (!(str = getStringDigits(7)));
            if (continueLoop("Are you sure?\n<1) - yes, 0) - no>"))
                database.customers[num].phoneNumber = str;
            break;
        case '6':
            system("cls");
            printf("Enter new card number for customer %u:\n>", num + 1);
            while (!(str = getStringDigits(16)));
            if (continueLoop("Are you sure?\n<1) - yes, 0) - no>"))
                database.customers[num].cardNumber = str;
            break;
        case '0':
            return false;
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

char customerMenu(char *str) {
    char option;
    do {
        system("cls");
        puts(str);
        puts("1)\t<Surname>");
        puts("2)\t<Name>");
        puts("3)\t<Patronymic>");
        puts("4)\t<Address>");
        puts("5)\t<Phone number>");
        puts("6)\t<Card number>");
        puts("0)\t<Back>");
        printf(">");
    } while (!(option = getOption('0', '6')));
    return option;
}

bool continueLoop(char *str) {
    char option;
    system("cls");
    do puts(str);
    while (!(option = getOption('0', '1')));
    return (option == '1') ? true : false;
}

//######################################################################################################################
// Remove customer from data base
//######################################################################################################################
struct customersInfo deleteCustomers(struct customersInfo database) {
    size_t num;
    do {
        system("cls");
        // Zero check
        if (amountZeroCheck(database.amount))
            return database;
        system("cls");
        do
            printf("Enter the number of the user you want to remove:\n>");
        while (!(num = compareWithAmount(database.amount)));
        if (!continueLoop("Are you sure?\n<1) - yes, 0) - no>"))
            continue;

        database = delete(database, num - 1);
    } while (continueLoop("continue removing customers from the database?\n<1) - yes, 0) - no>"));
    return database;
}

struct customersInfo delete(struct customersInfo database, size_t deleteNum) {
    // Remember last struct
    struct customer buffer = database.customers[database.amount - 1];
    // If there is only 1 struct
    if (database.amount == 1) {
        database.amount = 0;
        freeStruct(database.customers, deleteNum);
        free(database.customers);
        database.customers = NULL;
        return database;
    }
    // Reallocation check
    struct customer *temp;
    if (!(temp = (struct customer *) realloc(database.customers, (database.amount - 1) * sizeof(struct customer)))) {
        puts("Not enough memory!");
        system("pause>0");
        return database;
    }
    database.customers = temp;
    database.amount--;
    // If we're deleting last struct
    if (database.amount == deleteNum)
        return database;
    // Transfer information from struct to struct
    size_t i;
    for (i = deleteNum; i < database.amount - 1; ++i)
        database.customers[i] = database.customers[i + 1];
    database.customers[i] = buffer;
    return database;
}

//######################################################################################################################
// Sort customers
//######################################################################################################################
void sortCustomers(struct customersInfo database) {
    // Zero check
    if (amountZeroCheck(database.amount))
        return;
    // Choose the sorting direction
    bool flag = continueLoop("<1 - ascending, 0 - descending>");
    switch (customerMenu("select an option to sort:")) {
        case '1':
            sortSurname(database, flag);
            break;
        case '2':
            sortName(database, flag);
            break;
        case '3':
            sortPatronymic(database, flag);
            break;
        case '4':
            sortAddress(database, flag);
            break;
        case '5':
            sortPhoneNumber(database, flag);
            break;
        case '6':
            sortCardNumber(database, flag);
            break;
        case '0':
            return;
    }
}

void swap(struct customer *x, struct customer *y) {
    struct customer temp = *x;
    *x = *y;
    *y = temp;
}

void sortSurname(struct customersInfo database, bool flag) {
    for (size_t i = 0; i < database.amount; i++)
        for (size_t j = i + 1; j < database.amount; j++)
            if ((flag && strcmp(database.customers[i].surname, database.customers[j].surname) > 0) ||
                (!flag && strcmp(database.customers[i].surname, database.customers[j].surname) < 0))
                swap(&database.customers[i], &database.customers[j]);
}

void sortName(struct customersInfo database, bool flag) {
    for (size_t i = 0; i < database.amount; i++)
        for (size_t j = i + 1; j < database.amount; j++)
            if ((flag && strcmp(database.customers[i].name, database.customers[j].name) > 0) ||
                (!flag && strcmp(database.customers[i].name, database.customers[j].name) < 0))
                swap(&database.customers[i], &database.customers[j]);
}

void sortPatronymic(struct customersInfo database, bool flag) {
    for (size_t i = 0; i < database.amount; i++)
        for (size_t j = i + 1; j < database.amount; j++)
            if ((flag && strcmp(database.customers[i].patronymic, database.customers[j].patronymic) > 0) ||
                (!flag && strcmp(database.customers[i].patronymic, database.customers[j].patronymic) < 0))
                swap(&database.customers[i], &database.customers[j]);
}

void sortAddress(struct customersInfo database, bool flag) {
    // My Precious<3
    for (size_t i = 0; i < database.amount; i++)
        for (size_t j = i + 1; j < database.amount; j++) {
            if ((flag && strcmp(database.customers[i].address.street, database.customers[j].address.street) > 0) ||
                (!flag && strcmp(database.customers[i].address.street, database.customers[j].address.street) < 0))
                swap(&database.customers[i], &database.customers[j]);
            if (!strcmp(database.customers[i].address.street, database.customers[j].address.street)) {
                if ((flag && database.customers[i].address.houseNumber > database.customers[j].address.houseNumber) ||
                    (!flag && database.customers[i].address.houseNumber < database.customers[j].address.houseNumber))
                    swap(&database.customers[i], &database.customers[j]);
                if (database.customers[i].address.houseNumber == database.customers[j].address.houseNumber) {
                    if (database.customers[i].address.flatNumber == database.customers[j].address.flatNumber)
                        continue;
                    if ((flag && database.customers[i].address.flatNumber > database.customers[j].address.flatNumber) ||
                        (!flag && database.customers[i].address.flatNumber < database.customers[j].address.flatNumber))
                        swap(&database.customers[i], &database.customers[j]);
                }

            }

        }
}

void sortPhoneNumber(struct customersInfo database, bool flag) {
    for (size_t i = 0; i < database.amount; i++)
        for (size_t j = i + 1; j < database.amount; j++)
            if ((flag && strcmp(database.customers[i].phoneNumber, database.customers[j].phoneNumber) > 0) ||
                (!flag && strcmp(database.customers[i].phoneNumber, database.customers[j].phoneNumber) < 0))
                swap(&database.customers[i], &database.customers[j]);
}

void sortCardNumber(struct customersInfo database, bool flag) {
    for (size_t i = 0; i < database.amount; i++)
        for (size_t j = i + 1; j < database.amount; j++)
            if ((flag && strcmp(database.customers[i].cardNumber, database.customers[j].cardNumber) > 0) ||
                (!flag && strcmp(database.customers[i].cardNumber, database.customers[j].cardNumber) < 0))
                swap(&database.customers[i], &database.customers[j]);
}

//######################################################################################################################
// Find customers
//######################################################################################################################
void searchCustomers(struct customersInfo database) {
    // Zero check
    if (amountZeroCheck(database.amount))
        return;
    // Get key word for searching
    char *keyWords = NULL;
    size_t keyWordsLen = 0;
    do {
        system("cls");
        puts("Enter part of the word to search:");
        puts("<example: *ab*c*>");
        putchar('>');
    } while (!(keyWords = getKeyWord(&keyWordsLen)));
    // Need for atoi
    char *temp = NULL;
    if (!(temp = (char *) realloc(temp, 4 * sizeof(char)))) {
        puts("Not enough memory!");
        system("pause>0");
        return;
    }
    // Flag is checking for zero matches
    bool flag = true;
    for (size_t i = 0; i < database.amount; ++i) {
        if (compare(keyWords, keyWordsLen, database.customers[i].surname)) {
            flag = false;
            printf("Match with the %u customer's surname\t%s\n", i + 1, database.customers[i].surname);
        }
        if (compare(keyWords, keyWordsLen, database.customers[i].name)) {
            flag = false;
            printf("Match with the %u customer's name\t%s\n", i + 1, database.customers[i].name);
        }
        if (compare(keyWords, keyWordsLen, database.customers[i].patronymic)) {
            flag = false;
            printf("Match with the %u customer's patronymic\t%s\n", i + 1, database.customers[i].patronymic);
        }
        if (compare(keyWords, keyWordsLen, database.customers[i].address.street)) {
            flag = false;
            printf("Match with the %u customer's street\t%s\n", i + 1, database.customers[i].address.street);
        }
        if (compare(keyWords, keyWordsLen, _ultoa(database.customers[i].address.houseNumber, temp, 10))) {
            flag = false;
            printf("Match with the %u customer's house number\t%u\n", i + 1, database.customers[i].address.houseNumber);
        }
        if (compare(keyWords, keyWordsLen, _ultoa(database.customers[i].address.flatNumber, temp, 10))) {
            flag = false;
            printf("Match with the %u customer's flat number\t%u\n", i + 1, database.customers[i].address.flatNumber);
        }
        if (compare(keyWords, keyWordsLen, database.customers[i].phoneNumber)) {
            flag = false;
            printf("Match with the %u customer's phone number\t%s\n", i + 1, database.customers[i].phoneNumber);
        }
        if (compare(keyWords, keyWordsLen, database.customers[i].cardNumber)) {
            flag = false;
            printf("Match with the %u customer's card number\t%s\n", i + 1, database.customers[i].cardNumber);
        }
    }
    if (flag)
        puts("No matches!");
    system("pause>0");
}

char *getKeyWord(size_t *keyWordLen) {
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
    *keyWordLen = i;
    return str;
}

bool compare(char *keyWords, size_t keyWordsLen, char *str) {
    char *pos = NULL, *lastPos;
    size_t i = 0, j;
    // if no * on first place
    if (keyWords[0] != '\0')
        for (; keyWords[i] != '\0' && keyWords[i] != '\n'; i++)
            if (keyWords[i] != str[i])
                return false;
    // equality check
    if (keyWords[0] != '\0' && keyWords[keyWordsLen] == '\n' && strlen(str) == keyWordsLen)
        return true;
    // if no * on last place
    if (keyWords[keyWordsLen] == '\n')
        for (keyWordsLen--, j = 1; keyWordsLen >= 0 && keyWords[keyWordsLen] != '\0'; keyWordsLen--, j++)
            if (keyWords[keyWordsLen] != str[strlen(str) - j])
                return false;
    // check main body
    while (i <= keyWordsLen) {
        if (keyWords[i] == '\0') {
            i++;
            continue;
        }
        lastPos = pos;
        if (!(pos = strstr(str, keyWords + i)))
            return false;
        if (pos < lastPos)
            return false;
        i += strlen(keyWords + i);
        str = pos + strlen(keyWords + i);
    }
    return true;
}
//######################################################################################################################
bool amountZeroCheck(size_t amount) {
    if (amount == 0) {
        puts("There is no customer information in the database!");
        system("pause>0");
        return true;
    }
    return false;
}

void freeStruct(struct customer *customers, size_t num) {
    free(customers[num].surname);
    free(customers[num].name);
    free(customers[num].patronymic);
    free(customers[num].address.street);
    free(customers[num].phoneNumber);
    free(customers[num].cardNumber);
}
//######################################################################################################################