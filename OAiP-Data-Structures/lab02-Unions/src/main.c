#include "functions.h"

int main(void) {
    database database;
    database.workers = NULL;
    database.amount = 0;
    do {
        switch (menu()) {
            case '1':
                database = add(database);
                break;
            case '2':
                show(database);
                break;
            case '3':
                find(database);
                break;
            case '4':
                database = delete(database);
                break;
            case '5':
                findMinSalary(database);
                break;
            case '6':
                database = deleteWorkerWithDate(database);
                break;
            case '0':
                for (size_t i = database.amount - 1; i >= 0; --i)
                    free(database.workers[i].surname);
                free(database.workers);
                return 0;
        }
    } while (true);
}