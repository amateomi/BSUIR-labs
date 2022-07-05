#include "functions.h"

int main(void) {
    tree_t *root = NULL;

    // Destroy Clion buffer for debug mode
    setbuf(stdout, NULL);
    do
        switch (menu()) {
            case '1':
                // Add words to tree
                add(&root);
                break;

            case '2':
                // Show words from tree
                show(root);
                break;

            case '3':
                // Show words from tree (recursive)
                if (emptyCheckTreeAndPrintMessage(root))
                    break;
                printf("English\tRussian\n");
                showRecursive(root);
                break;

            case '4':
                // Delete specific word
                deleteWord(&root);
                break;

            case '5':
                // Clear tree
                if (emptyCheckTreeAndPrintMessage(root))
                    break;
                clear(&root);
                break;

            case '6':
                // Output picture tree
                if (emptyCheckTreeAndPrintMessage(root))
                    break;
                showAsTree(root, 0);
                break;

            case '0':
                // Exit the program
                clear(&root);
                return EXIT_SUCCESS;
        }
    while (1);
}

