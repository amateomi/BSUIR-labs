#include <stdio.h>

#define SIZE 10

void clear_stdin(void) {
  int c;
  while ((c = getchar()) != '\n' && c != EOF);
}

float c_calculate(float x, int n);
void asm_calculate(void);

float array[SIZE];
int power;

int main(void) {
  printf("Enter %d float numbers:\n", SIZE);
  for (int i = 0; i < SIZE; ++i) {
    if (scanf("%f", array + i) != 1 || array[i] < 0.0) {
      fprintf(stderr, "Failed to read positive double number!\n");
      clear_stdin();
      --i;
    }
  }

  puts("Enter power:\n");
  while (scanf("%i", &power) != 1 || power < 0) {
    fprintf(stderr, "Failed to read not negative power!\n");
    clear_stdin();
  }

  puts("C calculations:");
  for (int i = 0; i < SIZE; ++i) {
    printf("%f\t", c_calculate(array[i], power));
  }
  putchar('\n');

  puts("Asm calculations:");
  asm_calculate();
  for (int i = 0; i < SIZE; ++i) {
    printf("%f\t", array[i]);
  }
  putchar('\n');
}

float c_calculate(float x, int n) {
  if (n == 0) {
    return 1.0f;
  }
  float res = x;

  for (int i = 0; i < n - 1; ++i) {
    res *= x;
  }
  return res;
}

void asm_calculate(void) {
  short cr_register;

  // x**n = 2**(log2(x**n)) = 2**(n*log2(x)), let y = n*log2(x)
  asm volatile (
      "finit\n"

      // This algorithm requires rounding to zero
      "fstcw %2\n"
      "or %2, 0x0C00\n"
      "fldcw %2\n"

      // Load array head to rbx
      "lea rbx, %0\n"
      // Load array size
      "mov rcx, 10\n"

      "again:\n"

      // st1 = n, st0 = x
      "fild dword ptr %1\n"
      "fld dword ptr [rbx]\n"

      // st0 = n*log2(x) = y
      "fyl2x\n"

      // st1 = y, st0 = 2**(y - trunc(y)) - 1
      "fld st(0)\n"
      "frndint\n"
      "fsubr st(0), st(1)\n"
      // mod(st0) must be not greater than 1, so all this mess because of it
      "f2xm1\n"

      // st1 = y, st0 = 2**(y - trunc(y))
      "fld1\n"
      "faddp\n"
      //           old st0           truncated st1
      // st0 = 2**(y - trunc(y)) * 2**(trunc(y)) = 2**(y)
      "fscale\n"

      // Remove y from st1
      "fxch st(1)\n"
      "fstp st\n"

      // Save number to array
      "fstp dword ptr [rbx]\n"

      // Shift array pointer
      "add rbx, 4\n"
      "loop again\n"

      "fwait\n"
      : "=m" (array) // Output
      : "m" (power), "m" (cr_register) // Input
      );
}
