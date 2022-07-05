#include "functions.h"

//#####################################################################################################################
bool emptyCheckTreeAndPrintMessage(tree_t *root) {
    bool isEmpty;
    if ((isEmpty = isEmptyRoot(root)))
        puts("Tree is empty!");
    return isEmpty;
}

bool isEmptyRoot(tree_t *root) {
    return (root == NULL) ? true : false;
}

tree_t *addNode(tree_t *node, dictionary newWord) {
    if (isEmptyRoot(node)) {
        // Allocate memory
        if (!(node = (tree_t *) malloc(sizeof(tree_t)))) {
            PRINT_ALLOCATION_ERROR_MESSAGE
            return NULL;
        }
        node->data = newWord;
        node->left = node->right = NULL;
    }
        // Left subtree
    else if (strcmp(newWord.englishWord, node->data.englishWord) < 0)
        node->left = addNode(node->left, newWord);
        // Right subtree
    else if (strcmp(newWord.englishWord, node->data.englishWord) > 0)
        node->right = addNode(node->right, newWord);
    else
        puts("This word is already in the tree!");
    return node;
}

//#####################################################################################################################
char getOption(const char leftBorder, const char rightBorder) {
    char option, newRow;
    rewind(stdin);
    if (scanf("%c%c", &option, &newRow) != 2 || newRow != '\n' || option < leftBorder || option > rightBorder) {
        PRINT_INPUT_ERROR_MESSAGE
        return '\0';
    }
    return option;
}

bool getEnglishWord(char word[WORD_SIZE]) {
    size_t i;
    rewind(stdin);
    for (i = 0; (word[i] = (char) getchar()) != '\n'; ++i) {
        // Array overflow check
        if (i == WORD_SIZE) {
            PRINT_OVERFLOW_MESSAGE
            return false;
        }
        // Letter check
        if ((word[i] < 'A' || word[i] > 'Z') && (word[i] < 'a' || word[i] > 'z')) {
            puts("Invalid input! Use only english letters!");
            return false;
        }
        // Convert letters to the correct case
        if (word[i] >= 'A' && word[i] <= 'Z')
            word[i] = (char) (word[i] + ('a' - 'A'));
    }
    word[i] = '\0';
    return (word[0] == '\0') ? false : true;
}

bool getRussianWord(char word[WORD_SIZE]) {
    size_t i;
    rewind(stdin);
    for (i = 0; i < WORD_SIZE && (word[i] = (char) getchar()) != '\n'; ++i) {
        // Array overflow check
        if (i == WORD_SIZE) {
            PRINT_OVERFLOW_MESSAGE
            return false;
        }
        // Letter check
        //        if ((word[i] > (-81) && word[i] < (-32)) || word[i] > (-17))
        //        {
        //            puts("Invalid input! Use only russian letters!");
        //            return false;
        //        }
    }
    word[i] = '\0';
    return (word[0] == '\0') ? false : true;
}

dictionary getWords(void) {
    dictionary newWord;
    do
        printf("Enter english word:\n>");
    while (!getEnglishWord(newWord.englishWord));
    do
        printf("Enter russian word:\n>");
    while (!getRussianWord(newWord.russianWord));
    return newWord;
}

bool getUnsigned(size_t *number) {
    char newRow;

    // Negative number check
    rewind(stdin);
    if ((newRow = (char) getc(stdin)) == '-') {
        puts("Negative numbers cannot be entered! Try again.");
        return false;
    }
    ungetc(newRow, stdin);

    if (scanf("%u%c", number, &newRow) != 2 || newRow != '\n') {
        PRINT_INPUT_ERROR_MESSAGE
        return false;
    }

    return number;
}

void add(tree_t **root) {
    size_t wordAmount;

    // Get amount of new words
    do
        printf("How many words do you want to enter?\n>");
    while (!getUnsigned(&wordAmount));
    // Get and add new words
    for (int i = 0; i < wordAmount; ++i) {
        printf("New words num %i:\n", i + 1);
        *root = addNode(*root, getWords());
    }
}

//#####################################################################################################################
char menu(void) {
    char option;
    do {
        puts("Select an option from the list below:");
        puts("1)\t<Add words to tree>");
        puts("2)\t<Show words from tree>");
        puts("3)\t<Show words from tree (recursive)>");
        puts("4)\t<Delete specific word>");
        puts("5)\t<Clear tree>");
        puts("0)\t<Exit the program>");
        puts("Additional feature:");
        puts("6)\t<Print tree picture>");
        putchar('>');
    } while (!(option = getOption('0', '6')));
    return option;
}

void showRecursive(tree_t *node) {
    if (node) {
        printf("%s\t\t%s\n", node->data.englishWord, node->data.russianWord);
        if (node->left)
            showRecursive(node->left);
        if (node->right)
            showRecursive(node->right);
    }
}

// Stack for non-recursive printing tree
typedef struct stack_t {
    tree_t *node;
    struct stack_t *next;
} stack_t;

void stackPush(stack_t **head, tree_t *node) {
    stack_t *newHead;
    if (!(newHead = (stack_t *) malloc(sizeof(stack_t)))) {
        PRINT_ALLOCATION_ERROR_MESSAGE
        return;
    }
    newHead->next = *head;
    newHead->node = node;
    *head = newHead;
}

tree_t *stackPop(stack_t **head) {
    stack_t *prevHead;
    tree_t *treeNode;

    if (*head == NULL)
        return NULL;

    prevHead = *head;
    *head = (*head)->next;
    treeNode = prevHead->node;
    free(prevHead);
    return treeNode;
}

void show(tree_t *node) {
    stack_t *stack = NULL;

    if (emptyCheckTreeAndPrintMessage(node))
        return;
    else
        printf("English\tRussian\n");

    while (node || stack) {
        if (node) {
            stackPush(&stack, node);
            node = node->left;
        } else {
            node = stackPop(&stack);
            printf("%s\t\t%s\n", node->data.englishWord, node->data.russianWord);
            node = node->right;
        }
    }
}


void showAsTree(tree_t *node, int space) {
    // Stop when NULL
    if (node == NULL)
        return;

    // Increase distance between levels
    space += WORD_LENGTH;

    // Go right first
    showAsTree(node->right, space);

    // Print current node after spaces
    printf("\n");
    for (int i = WORD_LENGTH; i < space; i++)
        printf(" ");
    printf("%s\n", node->data.englishWord);

    // Go left
    showAsTree(node->left, space);
}

//#####################################################################################################################
void clear(tree_t **node) {
    if (isEmptyRoot(*node))
        return;
    if ((*node)->left)
        clear(&((*node)->left));
    if ((*node)->right)
        clear(&((*node)->right));
    free(*node);
    *node = NULL;
}

typedef struct relatedNodes_t {
    tree_t *node;
    tree_t *parent;
} relatedNodes_t;

relatedNodes_t findNodeToDeleteAndReturnRelatedNodes(tree_t *node, char *wordToDelete) {
    static relatedNodes_t relatedNodes = {NULL, NULL};
    static bool noMatches;

    noMatches = false;

    if (strcmp(wordToDelete, node->data.englishWord) < 0) {
        // Move to left subtree if not NULL
        if (node->left) {
            relatedNodes.parent = node;
            findNodeToDeleteAndReturnRelatedNodes(node->left, wordToDelete);
        }
            // Else no word in tree
        else
            noMatches = true;

    } else if (strcmp(wordToDelete, node->data.englishWord) > 0) {
        // Move to right subtree if not NULL
        if (node->right) {
            relatedNodes.parent = node;
            findNodeToDeleteAndReturnRelatedNodes(node->right, wordToDelete);
        }
            // Else no word in tree
        else
            noMatches = true;
    } else
        relatedNodes.node = node;
    // Return tree root if no word in tree
    if (noMatches == true)
        relatedNodes.node = relatedNodes.parent = NULL;
    // Return previous node in tree
    return relatedNodes;
}

relatedNodes_t findMaxNodeAndReturnRelatedNodes(tree_t *node) {
    static relatedNodes_t max = {NULL, NULL};

    // Go right as long as the right node exists
    if (node->right) {
        max.parent = node;
        findMaxNodeAndReturnRelatedNodes(node->right);
    } else
        max.node = node;

    return max;
}

void deleteWord(tree_t **root) {
    relatedNodes_t removedNode;
    char wordToDelete[WORD_SIZE];

    if (emptyCheckTreeAndPrintMessage(*root))
        return;
    // Get word to delete
    do
        printf("Enter english word to delete:\n>");
    while (!getEnglishWord(wordToDelete));
    // Finding position in tree, NULL - no word, root - if node == *root
    removedNode = findNodeToDeleteAndReturnRelatedNodes(*root, wordToDelete);
    // Word match check
    if (!removedNode.node && !removedNode.parent) {
        puts("No such word in the tree!");
        return;
    }
    puts("Remove is done!");
    // Different situations for deleted node
    // 1. No childs
    if (!removedNode.node->left && !removedNode.node->right) {
        // Root case
        if (removedNode.node == *root) {
            free(*root);
            *root = NULL;
        }
            // Left case
        else if (removedNode.node == removedNode.parent->left) {
            free(removedNode.parent->left);
            removedNode.parent->left = NULL;
        }
            // Right case
        else if (removedNode.node == removedNode.parent->right) {
            free(removedNode.parent->right);
            removedNode.parent->right = NULL;
        }
    }
        // 2. Only left child
    else if (removedNode.node->left && !removedNode.node->right) {
        // Root case
        if (removedNode.node == *root) {
            removedNode.parent = *root;
            *root = removedNode.node->left;
            free(removedNode.parent);
        }
            // Left case
        else if (removedNode.node == removedNode.parent->left) {
            removedNode.parent->left = removedNode.node->left;
            free(removedNode.node);
        }
            // Right case
        else if (removedNode.node == removedNode.parent->right) {
            removedNode.parent->right = removedNode.node->left;
            free(removedNode.node);
        }
    }
        // 3. Only right child
    else if (!removedNode.node->left && removedNode.node->right) {
        // Root case
        if (removedNode.node == *root) {
            removedNode.parent = *root;
            *root = removedNode.node->right;
            free(removedNode.parent);
        }
            // Left case
        else if (removedNode.node == removedNode.parent->left) {
            removedNode.parent->left = removedNode.node->right;
            free(removedNode.node);
        }
            // Right case
        else if (removedNode.node == removedNode.parent->right) {
            removedNode.parent->right = removedNode.node->right;
            free(removedNode.node);
        }
    }
        // 4. Two childs
    else if (removedNode.node->left && removedNode.node->right) {
        // 1) Finding a node to replace in left subtree
        relatedNodes_t maxNode = findMaxNodeAndReturnRelatedNodes(removedNode.node->left);
        // if max stay left
        if (maxNode.parent == NULL) {
            maxNode.parent = removedNode.node;
            if (removedNode.node == *root) {
                maxNode.node->right = maxNode.parent->right;
                *root = maxNode.node;
                free(maxNode.parent);
            } else if (removedNode.node == removedNode.parent->left) {
                maxNode.node->right = removedNode.node->right;
                removedNode.parent->left = removedNode.node->left;
                free(removedNode.node);
            } else if (removedNode.node == removedNode.parent->right) {
                maxNode.node->right = removedNode.node->right;
                removedNode.parent->right = removedNode.node->left;
                free(removedNode.node);
            }
            return;
        }
        // 2) MaxNode right pointer = removed node right pointer
        maxNode.node->right = removedNode.node->right;
        // Check for single node in left subtree
        // 3) Set parent maxNode right pointer on maxNode left subtree
        maxNode.parent->right = maxNode.node->left;
        // 4) MaxNode left pointer = parent maxNode
        maxNode.node->left = removedNode.node->left;
        // 5) Set parent removedNode pointer on maxNode
        // Root case
        if (removedNode.node == *root)
            *root = maxNode.node;
            // Left case
        else if (removedNode.node == removedNode.parent->left)
            removedNode.parent->left = maxNode.node;
            // Right case
        else if (removedNode.node == removedNode.parent->right)
            removedNode.parent->right = maxNode.node;
        // 6) Free removedNode
        free(removedNode.node);
    }
}
//#####################################################################################################################