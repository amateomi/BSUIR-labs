#include "functions.h"

static FILE *textFile, *binFile;
static char *fileTextName = NULL, *fileBinName = NULL;

// true - data successfully got, false - invalid data
bool getDepartmentCodeFromTextFile(ull *departmentCode) {
    char c;
    if (fscanf(textFile, "%llu%c", departmentCode, &c) != 2 || c != '\n') {
        puts("Invalid data for department code!");
        system("pause>0");
        return false;
    }
    return true;
}

bool getLastNameFromTextFile(char **lastName) {
    // Surname check
    size_t i;
    char c;
    for (i = 0; (c = (char) fgetc(textFile)) != '\n'; ++i) {
        // Letter check
        if ((c < 'A' || c > 'Z') && (c < 'a' || c > 'z')) {
            puts("Invalid characters for lastName!");
            system("pause>0");
            return false;
        }
        // Reallocation check
        if (!((*lastName) = (char *) realloc((*lastName), i + 2))) {
            puts("Not enough memory!");
            system("pause>0");
            return false;
        }
        // Convert letters to the correct case
        (c >= 'A' && c <= 'Z') ? (*lastName)[i] = (char) (c + ('a' - 'A')) : ((*lastName)[i] = c);
    }
    // Empty pointer check
    if ((*lastName) != NULL)
        (*lastName)[i] = '\0';
    else {
        puts("No lastName in file!");
        system("pause>0");
        return false;
    }
    return true;
}

bool getUnionInfoTypeFromTextFile(bool *type) {
    // Info type for union check
    char c;
    if (((c = (char) fgetc(textFile)) != '1' && c != '0') || (char) fgetc(textFile) != '\n') {
        puts("Invalid data for info type!");
        system("pause>0");
        return false;
    }
    *type = (c == '1') ? true : false;
    return true;
}

bool getDateFromTextFile(char date[8]) {
    // Date check
    for (size_t i = 0; i < 8; i++)
        if (((i == 2 || i == 4) && (char) fgetc(textFile) != '.')
            || ((date[i] = (char) fgetc(textFile)) < '0' || date[i] > '9')) {
            puts("Invalid characters for date!");
            system("pause>0");
            return false;
        }
    if ((date[2] == '1' && date[3] > '2')
        || date[2] > '1' || date[0] > '3'
        || (date[0] == '3' && date[1] > 1)) {
        puts("Invalid data! Such date doesn't exist!");
        system("pause>0");
        return false;
    }
    return true;
}

bool getSalaryFromTextFile(ull *salary) {
    // Salary check
    if (fscanf(textFile, "%llu", salary) != 1) {
        puts("Invalid data for salary!");
        system("pause>0");
        return false;
    }
    return true;
}

bool getDepartmentCodeFromBinFile(ull *departmentCode) {
    // Department code check
    if (feof(binFile))
        return false;
    if (fread(departmentCode, sizeof(ull), 1, binFile) != 1) {
        puts("Invalid data for department code!");
        system("pause>0");
        return false;
    }
    return true;
}

bool getLastNameFromBinFile(char **lastName, char *c) {
    // Surname check
    size_t i;
    for (i = 0; fread(c, sizeof(char), 1, binFile) == 1; ++i) {
        // Read until letter
        if ((*c < 'A' || *c > 'Z') && (*c < 'a' || *c > 'z'))
            break;
        // Reallocation check
        if (!((*lastName) = (char *) realloc((*lastName), i + 2))) {
            puts("Not enough memory!");
            system("pause>0");
            return false;
        }
        // Convert letters to the correct case
        (*c >= 'A' && *c <= 'Z') ? (*lastName)[i] = (char) (*c + ('a' - 'A')) : ((*lastName)[i] = *c);
    }
    // Empty pointer check
    if ((*lastName) != NULL)
        (*lastName)[i] = '\0';
    else {
        puts("No lastName in file!");
        system("pause>0");
        return false;
    }
    return true;
}

bool getDateFromBinFile(char date[8]) {
    // Date check
    if (fread(date, sizeof(char), 8, binFile) != 8) {
        puts("No date in file!");
        system("pause>0");
        return false;
    }
    for (size_t i = 0; i < 8; ++i) {
        if ((date[2] == '1' && date[3] > '2')
            || date[2] > '1' || date[0] > '3'
            || (date[0] == '3' && date[1] > 1)) {
            puts("Invalid data! Such date doesn't exist!");
            system("pause>0");
            return false;
        }
    }
    return true;
}

bool getSalaryFromBinFile(ull *salary) {
    // Salary check
    if (fread(salary, sizeof(ull), 1, binFile) != 1) {
        puts("Invalid data for salary!");
        system("pause>0");
        return false;
    }
    return true;
}

void load(queue_t *queue) {
    worker_t worker;
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
                return;
            }
        } while (1);
        // Get data from file and load it into queue
        do {
            worker.lastName = NULL;
            if (!getDepartmentCodeFromTextFile(&worker.departmentCode)
                || !getLastNameFromTextFile(&worker.lastName)
                || !getUnionInfoTypeFromTextFile(&worker.infoType))
                return;
            if (worker.infoType == false)
                if (!getDateFromTextFile(worker.info.date))
                    return;
            if (worker.infoType == true)
                if (!getSalaryFromTextFile(&worker.info.salary))
                    return;
            if (!push(queue, worker))
                return;
        } while (fgetc(textFile) == '\n');
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
                return;
            }
        } while (1);
        do {
            worker.lastName = NULL;
            char infoType;
            if (!getDepartmentCodeFromBinFile(&worker.departmentCode)
                || !getLastNameFromBinFile(&worker.lastName, &infoType))
                return;
            // Info type for union check
            if (infoType != 1 && infoType != 0) {
                puts("Invalid data for info type!");
                system("pause>0");
                return;
            }
            worker.infoType = (infoType == 1) ? true : false;
            if (worker.infoType == false)
                if (!getDateFromBinFile(worker.info.date))
                    return;
            if (worker.infoType == true)
                if (!getSalaryFromBinFile(&worker.info.salary))
                    return;
            push(queue, worker);
            fread(&infoType, sizeof(char), 1, binFile);
            if (feof(binFile))
                break;
            else
                fseek(binFile, -1, SEEK_CUR);
        } while (1);
        fclose(binFile);
    }
}

void save(const queue_t queue) {
    // Zero check
    if (isEmpty(queue)) {
        puts("Database is empty!");
        system("pause>0");
        return;
    }
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
    for (node_t *node = queue.head; node != queue.tail->next; node = node->next) {
        // Add data from queue into text file
        if (fileType) {
            fprintf(textFile, "%llu\n", node->worker.departmentCode);
            fprintf(textFile, "%s\n", node->worker.lastName);
            fprintf(textFile, "%i\n", node->worker.infoType);
            if (!node->worker.infoType)
                for (size_t i = 0; i < 8; ++i) {
                    if (i == 2 || i == 4)
                        fputc('.', textFile);
                    fputc(node->worker.info.date[i], textFile);
                }
            else
                fprintf(textFile, "%llu", node->worker.info.salary);
            if (node != queue.tail)
                fputc('\n', textFile);
            else break;
        }
            // Add data from stack into bin file
        else {
            fwrite(&node->worker.departmentCode, sizeof(ull), 1, binFile);
            fwrite(node->worker.lastName, sizeof(char), strlen(node->worker.lastName), binFile);
            fwrite(&node->worker.infoType, sizeof(bool), 1, binFile);
            if (!node->worker.infoType)
                fwrite(node->worker.info.date, sizeof(char), 8, binFile);
            else
                fwrite(&node->worker.info.salary, sizeof(ull), 1, binFile);
        }
    }
    fclose(textFile);
    fclose(binFile);
}

