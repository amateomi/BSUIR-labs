#include "functions.h"

int main(void) {
    FILE *file;
    char *fileTextName, *fileBinName;
    // Get name for text file
    do {
        system("cls");
        printf("Enter name for text file:\n");
        putchar('>');
    } while (!(fileTextName = getFileName()));
    system("cls");
    // Create text file or just open existing one
    if (!(file = fopen(strcat(fileTextName, ".txt"), "a"))) {
        printf("Unable to open text file!");
        system("pause>0");
        return 1;
    }
    fclose(file);
    // Get name for binary file
    system("cls");
    do {
        system("cls");
        printf("Enter name for binary file:\n");
        putchar('>');
    } while (!(fileBinName = getFileName()));
    system("cls");
    // Create binary file or just open existing one
    if (!(file = fopen(strcat(fileBinName, ".bin"), "ab"))) {
        printf("Unable to open binary file!");
        system("pause>0");
        return 1;
    }
    fclose(file);
    do {
        switch (menu()) {
            case '1':
                // Add numbers to files
                addNumbers(fileTextName, fileBinName);
                break;
            case '2':
                // Show numbers from files
                showNumbers(fileTextName, fileBinName);
                break;
            case '3':
                // Show max numbers from text file
                findMaxNumber(fileTextName);
                break;
            case '4':
                // Reverse number in text file
                reverseNumberWithIndex(fileTextName);
                break;
            case '5':
                // Count average mark for sportsmen in bin file
                countAverageMark(fileBinName);
                break;
            case '6':
                // Swap max and min values in bin file
                swapMaxAndMin(fileBinName);
                break;
            case '0':
                // Exit the program
                return 0;
        }
    } while (true);
}