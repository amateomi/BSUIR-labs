#include "functions.h"

int main(void) {
    struct stack *head = NULL;
    uploadFromFile(&head);
    do {
        switch (menu()) {
            case '1':
                // Add employee
                addElements(&head);
                break;
            case '2':
                // Show employee
                showElements(head);
                break;
            case '3':
                // Find employee
                findElements(head);
                break;
            case '4':
                // Delete employee
                deleteElements(&head);
                break;
            case '5':
                // Save to files
                saveToTextFile(head);
                break;
            case '0':
                // Exit the program
                while (head)
                    delete(&head, 1);
                return 0;
        }
    } while (true);
}
