#include <stdio.h>

int main() {
    int n;

    // Get amount sequence members.
    printf("Enter n:");
    scanf("%i", &n);

    int amount = 0;
    // We cannot initialize the sum because
    // the sequence may not have a suitable member.
    int sum;

    int sequenceMember;
    for (int i = 0; i < n; ++i) {
        // Get next member of the sequence.
        printf("Enter sequence term:");
        scanf("%i", &sequenceMember);

        // If member suitable by conditions.
        if ((sequenceMember % 5 == 0) && (sequenceMember % 7 != 0)) {
            amount++;

            if (amount == 1) {
                // Initialize sum if amount == 1.
                sum = sequenceMember;
            } else {
                sum += sequenceMember;
            }
        }
    }

    if (amount == 0) {
        printf("No member of the sequence matches the condition!\n");
    } else {
        printf("%i members of the sequence fit, their sum is %i.\n", amount, sum);
    }

    return 0;
}
