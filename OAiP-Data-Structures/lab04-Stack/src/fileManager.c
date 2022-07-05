#include "functions.h"

FILE *textFile, *binFile;
char *fileTextName = NULL, *fileBinName = NULL;

void uploadFromFile(struct stack **head) {
    if (getBoolPrintMassage("Load date into stack from file?\n1)\t<yes>\n0)\t<no>\n>")) {
        struct employee element;
        if (getBoolPrintMassage("Which file type you want to open?\n1)\t<text>\n0)\t<bin>\n>"))
            // Text
        {
            // Get file name
            do {
                system("cls");
                printf("Enter text file name:\n>");
                if (!(fileTextName = getFileName()))
                    continue;
                // Open existing text file
                if ((textFile = fopen(strcat(fileTextName, ".txt"), "r")))
                    break;
                else {
                    printf("Unable to open text file!");
                    system("pause>0");
                    continue;
                }
            } while (1);
            // Get data from file and upload it into stack
            char c;
            do {
                // Department code check
                if (fscanf(textFile, "%llu%c", &element.departmentCode, &c) != 2 || c != ' ') {
                    puts("Invalid data for department code!");
                    system("pause>0");
                    exit(1);
                }
                // Surname check
                element.lastName = NULL;
                size_t i;
                for (i = 0; (c = (char) fgetc(textFile)) != ' '; ++i) {
                    // Letter check
                    if ((c < 'A' || c > 'Z') && (c < 'a' || c > 'z')) {
                        puts("Invalid characters for lastName!");
                        system("pause>0");
                        exit(1);
                    }
                    // Reallocation check
                    if (!(element.lastName = (char *) realloc(element.lastName, i + 2))) {
                        puts("Not enough memory!");
                        system("pause>0");
                        exit(1);
                    }
                    // Convert letters to the correct case
                    (c >= 'A' && c <= 'Z') ? element.lastName[i] = (char) (c + ('a' - 'A')) : (element.lastName[i] = c);
                }
                // Empty pointer check
                if (element.lastName)
                    element.lastName[i] = '\0';
                else {
                    puts("No lastName in file!");
                    system("pause>0");
                    exit(1);
                }
                // Info type for union check
                if (((c = (char) fgetc(textFile)) != '1' && c != '0') || (char) fgetc(textFile) != ' ') {
                    puts("Invalid data for info type!");
                    system("pause>0");
                    exit(1);
                }
                if (c == '0') {
                    element.infoType = false;
                    // Date check
                    for (i = 0; i < 8; i++)
                        if (((i == 2 || i == 4) && (char) fgetc(textFile) != '.')
                            || ((element.info.date[i] = (char) fgetc(textFile)) < '0' || element.info.date[i] > '9')) {
                            puts("Invalid characters for date!");
                            system("pause>0");
                            exit(1);
                        }
                    if ((element.info.date[2] == '1' && element.info.date[3] > '2')
                        || element.info.date[2] > '1' || element.info.date[0] > '3'
                        || (element.info.date[0] == '3' && element.info.date[1] > 1)) {
                        puts("Invalid data! Such date doesn't exist!");
                        system("pause>0");
                        exit(1);
                    }
                } else {
                    element.infoType = true;
                    // Salary check
                    if (fscanf(textFile, "%llu", &element.info.salary) != 1) {
                        puts("Invalid data for salary!");
                        system("pause>0");
                        exit(1);
                    }
                }
                if ((c = (char) fgetc(textFile)) != '\n' && !feof(textFile)) {
                    puts("Invalid data in the end!");
                    system("pause>0");
                    exit(1);
                }
                push(head, element);
            } while (c == '\n' && !feof(textFile));
            fclose(textFile);
        } else
            // Bin
        {
            // Get file name
            do {
                system("cls");
                printf("Enter bin file name:\n>");
                if (!(fileBinName = getFileName()))
                    continue;
                // Open existing bin file
                if ((binFile = fopen(strcat(fileBinName, ".bin"), "rb")))
                    break;
                else {
                    printf("Unable to open bin file!");
                    system("pause>0");
                    continue;
                }
            } while (1);
            char newLine;
            do {
                // Department code check
                if (fread(&element.departmentCode, sizeof(ull), 1, binFile) != 1) {
                    puts("Invalid data for department code!");
                    system("pause>0");
                    exit(1);
                }
                // Surname check
                char c;
                element.lastName = NULL;
                size_t i;
                for (i = 0; fread(&c, sizeof(char), 1, binFile) == 1; ++i) {
                    // Read until letter
                    if ((c < 'A' || c > 'Z') && (c < 'a' || c > 'z'))
                        break;
                    // Reallocation check
                    if (!(element.lastName = (char *) realloc(element.lastName, i + 2))) {
                        puts("Not enough memory!");
                        system("pause>0");
                        exit(1);
                    }
                    // Convert letters to the correct case
                    (c >= 'A' && c <= 'Z') ? element.lastName[i] = (char) (c + ('a' - 'A')) : (element.lastName[i] = c);
                }
                // Empty pointer check
                if (element.lastName)
                    element.lastName[i] = '\0';
                else {
                    puts("No lastName in file!");
                    system("pause>0");
                    exit(1);
                }
                // Info type for union check
                if (c != 1 && c != 0) {
                    puts("Invalid data for info type!");
                    system("pause>0");
                    exit(1);
                }
                if (c == 0) {
                    element.infoType = false;
                    // Date check
                    if (fread(element.info.date, sizeof(char), 8, binFile) != 8) {
                        puts("No date in file!");
                        system("pause>0");
                        exit(1);
                    }
                    for (size_t j = 0; j < 8; ++j) {
                        if ((element.info.date[2] == '1' && element.info.date[3] > '2')
                            || element.info.date[2] > '1' || element.info.date[0] > '3'
                            || (element.info.date[0] == '3' && element.info.date[1] > 1)) {
                            puts("Invalid data! Such date doesn't exist!");
                            system("pause>0");
                            exit(1);
                        }
                    }
                } else {
                    element.infoType = true;
                    // Salary check
                    if (fread(&element.info.salary, sizeof(ull), 1, binFile) != 1) {
                        puts("Invalid data for salary!");
                        system("pause>0");
                        exit(1);
                    }
                }
                push(head, element);
                if (fread(&newLine, sizeof(char), 1, binFile) != 1)
                    break;
                if (newLine != '\n') {
                    puts("Invalid data in end!");
                    system("pause>0");
                    exit(1);
                }
            } while (1);
            fclose(binFile);
        }
        reverse(head);
    }
}

void reverse(struct stack **head) {
    struct stack *originHead, *topElement, *botElement;
    // Set head on the bottom of the stack
    for (originHead = topElement = botElement = *head; botElement; botElement = botElement->next)
        if (botElement->next == NULL)
            *head = botElement;
    botElement = *head;
    do {
        // Reverse pointer
        if (topElement->next == botElement) {
            botElement->next = topElement;
            botElement = botElement->next;
            topElement = originHead;
            continue;
        }
        topElement = topElement->next;
    } while (botElement != originHead);
    botElement->next = NULL;
}

void saveToTextFile(struct stack *element) {
    // Zero check
    if (isEmpty(element))
        return;
    // Get file type
    bool fileType = getBoolPrintMassage("Enter file type to save in:\n1)\t<text>\n0)\t<bin>\n>");
    // Get file name
    char *fileName;
    do {
        system("cls");
        fileTextName ? printf("Data was loaded from %s\n", fileTextName)
                     : fileBinName ? printf("Data was loaded from %s\n", fileBinName)
                                   : printf("Data wasn't loaded from files\n");
        printf("Enter file name to save in:\n>");
    } while (!(fileName = getFileName()));
    fileType ? (textFile = fopen(strcat(fileName, ".txt"), "w")) : (binFile = fopen(strcat(fileName, ".bin"), "wb"));
    for (; element; element = element->next) {
        // Add data from stack into text file
        if (fileType) {
            fprintf(textFile, "%llu ", element->employee.departmentCode);
            fprintf(textFile, "%s ", element->employee.lastName);
            fprintf(textFile, "%i ", element->employee.infoType);
            if (!element->employee.infoType)
                for (size_t i = 0; i < 8; ++i) {
                    if (i == 2 || i == 4)
                        fputc('.', textFile);
                    fputc(element->employee.info.date[i], textFile);
                }
            else
                fprintf(textFile, "%llu", element->employee.info.salary);
            if (element->next)
                fputc('\n', textFile);
            else break;
        }
            // Add data from stack into bin file
        else {
            fwrite(&element->employee.departmentCode, sizeof(ull), 1, binFile);
            fwrite(element->employee.lastName, sizeof(char), strlen(element->employee.lastName), binFile);
            fwrite(&element->employee.infoType, sizeof(bool), 1, binFile);
            if (!element->employee.infoType)
                fwrite(element->employee.info.date, sizeof(char), 8, binFile);
            else
                fwrite(&element->employee.info.salary, sizeof(ull), 1, binFile);
            if (element->next) {
                char c = '\n';
                fwrite(&c, sizeof(char), 1, binFile);
            } else
                break;
        }
    }
    fclose(textFile);
    fclose(binFile);
}