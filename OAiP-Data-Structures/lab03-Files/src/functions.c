#include "functions.h"

// Print list of options and return one of them
char menu(void) {
    char option;
    do {
        system("cls");
        puts("select an option from the list below:");
        puts("1)\t<Add numbers to files>");
        puts("2)\t<Show numbers from files>\n");
        puts("3)\t<Show max numbers from text file>");
        puts("4)\t<Reverse number on text file>\n");
        puts("5)\t<Count average mark for sportsmen in bin file>");
        puts("6)\t<Swap max and min values in bin file>\n");
        puts("0)\t<Exit the program>");
        putchar('>');
    } while (!(option = getOption('0', '6')));
    return option;
}

// Return some option in specific range
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

// Add and show number from files
void addNumbers(char *fileTextName, char *fileBinName) {
    // Get file type to work with
    char fileType;
    do {
        system("cls");
        puts("Select the file type to add numbers:");
        puts("1)\t<Text file>");
        puts("2)\t<Bin file>");
        puts("3)\t<Text and bin files>");
        putchar('>');
    } while (!(fileType = getOption('1', '3')));
    // Get amount of new numbers
    int amount;
    do {
        system("cls");
        printf("How many numbers do you want to add? Enter the amount:\n>");
    } while ((amount = getUnsigned()) <= 0);
    // Add numbers into chosen file
    FILE *file;
    if (fileType == '1' || fileType == '3') {
        file = fopen(fileTextName, "a");
        for (size_t i = 0; i < amount; ++i)
            fprintf(file, "%i ", getInt(i, "Enter number %i in text file:\n>"));
        // Save problems check
        if (fclose(file) == EOF) {
            puts("Unable to close file correctly!");
            system("pause>0");
        }
    }
    if (fileType == '2' || fileType == '3') {
        file = fopen(fileBinName, "ab");
        for (size_t i = 0; i < amount; ++i) {
            int number = getInt(i, "Enter number %i in binary file:\n>");
            fwrite(&number, sizeof(int), 1, file);
        }
        // Save problems check
        if (fclose(file) == EOF) {
            puts("Unable to close file correctly!");
            system("pause>0");
        }
    }
}

void showNumbers(char *fileTextName, char *fileBinName) {
    system("cls");
    FILE *file;
    int numberFromFile;
    // Show number from text file
    file = fopen(fileTextName, "r");
    puts("Numbers from text file:");
    // Empty file check
    if (fscanf(file, "%i", &numberFromFile) == EOF)
        puts("Text file is empty!");
    else {
        printf("%i ", numberFromFile);
        while (fscanf(file, "%i", &numberFromFile) != EOF)
            printf("%i ", numberFromFile);
        putchar('\n');
    }
    fclose(file);
    // Show number from bin file
    file = fopen(fileBinName, "rb");
    puts("Numbers from bin file:");
    // Empty file check
    if (fread(&numberFromFile, sizeof(int), 1, file) != 1)
        puts("Binary file is empty!");
    else {
        printf("%i ", numberFromFile);
        while (fread(&numberFromFile, sizeof(int), 1, file) == 1)
            printf("%i ", numberFromFile);
        putchar('\n');
    }
    fclose(file);
    system("pause>0");
}

// Text file
void findMaxNumber(char *fileName) {
    system("cls");
    FILE *file = fopen(fileName, "r");
    int maxNumber;
    // Empty file check, get first number
    if (fscanf(file, "%i", &maxNumber) == EOF) {
        printf("There is no numbers!");
        system("pause>0");
        return;
    }
    // Find max number
    int numberFormFile;
    while (fscanf(file, "%i", &numberFormFile) != EOF)
        if (numberFormFile > maxNumber)
            maxNumber = numberFormFile;
    rewind(file);
    printf("Max number is %i\n", maxNumber);
    // Count max numbers
    size_t count = 0;
    while (fscanf(file, "%i", &numberFormFile) != EOF)
        if (numberFormFile == maxNumber)
            count++;
    printf("There is %u such numbers", count);
    system("pause>0");
    fclose(file);
}

void reverseNumberWithIndex(char *fileName) {
    FILE *file = fopen(fileName, "r+");
    // Get index to reverse
    int index;
    do {
        system("cls");
        printf("Enter index number to reverse:\n>");
    } while ((index = getUnsigned()) <= 0);
    int numToReverse;
    // Check for correctness of the index
    for (int i = 1; i <= index; ++i)
        if (fscanf(file, "%i", &numToReverse) == EOF) {
            system("cls");
            printf("There is no such index in the file!");
            system("pause>0");
            fclose(file);
            return;
        }
    // Reverse
    long numEnd = ftell(file) - 1;
    long numStart = ftell(file) - getNumberOfDigits(numToReverse);
    int digitFormEnd, digitFormStart;
    for (; numStart < numEnd; numStart++, numEnd--) {
        fseek(file, numEnd, SEEK_SET);
        digitFormEnd = fgetc(file);
        fseek(file, numStart, SEEK_SET);
        digitFormStart = fgetc(file);
        fseek(file, -1, SEEK_CUR);
        fputc(digitFormEnd, file);
        fseek(file, numEnd, SEEK_SET);
        fputc(digitFormStart, file);
    }
    fclose(file);
}

// Binary file
void countAverageMark(char *fileName) {
    system("cls");
    FILE *file = fopen(fileName, "rb");
    int minNumber, maxNumber, numberFromFile;
    // Empty file check
    if (fread(&numberFromFile, sizeof(int), 1, file) != 1) {
        puts("File is empty!");
        system("pause>0");
        fclose(file);
        return;
    }
    double averageMark = maxNumber = minNumber = numberFromFile;
    int amount = 1;
    // Find max and min numbers, store sum of all numbers in averageMark, count the amount of numbers
    while (fread(&numberFromFile, sizeof(int), 1, file) == 1) {
        if (numberFromFile < minNumber)
            minNumber = numberFromFile;
        if (numberFromFile > maxNumber)
            maxNumber = numberFromFile;
        averageMark += numberFromFile;
        amount++;
    }
    // Too few numbers check
    if (amount == 1 || amount == 2) {
        puts("Too few numbers to calculate the average mark!");
        system("pause>0");
        fclose(file);
        return;
    }
    // Calculate and print average mark
    averageMark = (averageMark - minNumber - maxNumber) / (amount - 2);
    printf("The average mark is %.4lf", averageMark);
    system("pause>0");
    fclose(file);
}

void swapMaxAndMin(char *fileName) {
    system("cls");
    FILE *file = fopen(fileName, "rb+");
    int maxNumber, minNumber, numberFromFile;
    // Empty file check
    if (fread(&numberFromFile, sizeof(int), 1, file) != 1) {
        printf("There is no numbers!");
        system("pause>0");
        return;
    }
    maxNumber = minNumber = numberFromFile;
    // Find max and min numbers
    while (fread(&numberFromFile, sizeof(int), 1, file) == 1) {
        if (numberFromFile > maxNumber)
            maxNumber = numberFromFile;
        if (numberFromFile < minNumber)
            minNumber = numberFromFile;
    }
    rewind(file);
    // Swap max and min numbers
    while (fread(&numberFromFile, sizeof(int), 1, file) == 1) {
        if (numberFromFile == minNumber) {
            fseek(file, -(long) sizeof(int), SEEK_CUR);
            fwrite(&maxNumber, sizeof(int), 1, file);
            fseek(file, 0, SEEK_CUR);
        } else if (numberFromFile == maxNumber) {
            fseek(file, -(long) sizeof(int), SEEK_CUR);
            fwrite(&minNumber, sizeof(int), 1, file);
            fseek(file, 0, SEEK_CUR);
        }
    }
    fclose(file);
}

// Return correct integer number
int getInt(size_t i, const char str[]) {
    int number;
    char newLine;
    do {
        system("cls");
        printf(str, i + 1);
        rewind(stdin);
        // Check
        if (scanf("%i%c", &number, &newLine) != 2 || newLine != '\n') {
            puts("Invalid input, try again!");
            system("pause>0");
            continue;
        } else return number;
    } while (true);
}

// Return positive integer or -1, if invalid input
int getUnsigned(void) {
    int number;
    char newLine;
    // Check
    rewind(stdin);
    if (scanf("%i%c", &number, &newLine) != 2 || newLine != '\n') {
        puts("Invalid input, try again!");
        system("pause>0");
        return -1;
    }
    return number;
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

long getNumberOfDigits(int number) {
    long i;
    if (number < 0)
        number *= -1;
    for (i = 1; (number /= 10) > 0; ++i);
    return i;
}