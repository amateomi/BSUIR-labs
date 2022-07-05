#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <unistd.h>

int main(int argc, char* argv[], char* envp[]) {
  if (argc != 2) {
    fprintf(stderr, "%d is invalid arguments amount, must be 2\n", argc);
    exit(EXIT_FAILURE);
  }

  printf("\033[1;32m"); // Green color

  puts("Child process data:");
  printf("Name: %s\n", argv[0]);
  printf("Pid: %d\n", getpid());
  printf("Ppid: %d\n", getppid());

  const int MAX_SIZE = 256;
  char      buffer[MAX_SIZE];
  FILE* fenvp = fopen(argv[1], "r");
  if (!fenvp) {
    perror("fenvp");
    exit(errno);
  }
  while (fgets(buffer, MAX_SIZE, fenvp) != NULL) {
    buffer[strcspn(buffer, "\n")] = '\0';
    printf("%s=%s\n", buffer, getenv(buffer));
  }

  printf("\033[0m"); // Turn off green color

  fclose(fenvp);
  exit(EXIT_SUCCESS);
}
