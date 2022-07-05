#include "functions.h"

int main(void) {
    queue_t queue = {NULL, NULL};
    do {
        switch (menu()) {
            case '1':
                // Add worker
                add(&queue);
                break;
            case '2':
                // Show worker
                show(queue);
                break;
            case '3':
                // Find worker
                search(queue);
                break;
            case '4':
                // Delete worker
                delete(&queue);
                break;
            case '5':
                // Load from file
                load(&queue);
                break;
            case '6':
                // Save to file
                save(queue);
                break;
            case '0':
                // Exit the program
                while (isEmpty(queue))
                    deleteNodeWithNumber(&queue, 1);
                return 0;
        }
    } while (true);
}