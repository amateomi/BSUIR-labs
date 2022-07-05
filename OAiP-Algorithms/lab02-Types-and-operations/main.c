#include <stdio.h>

int main() {
    double a, b, c;

    // Get our numbers.
    printf("Enter a:");
    scanf("%lf", &a);
    printf("Enter b:");
    scanf("%lf", &b);
    printf("Enter c:");
    scanf("%lf", &c);

    if (a >= b && b >= c) {
        a *= 2;
        b *= 2;
        c *= 2;
    } else {
        a = (a < 0) ? -a : a;
        b = (b < 0) ? -b : b;
        c = (c < 0) ? -c : c;
    }

    printf("a = %lf, b = %lf, c = %lf\n", a, b, c);

    return 0;
}
