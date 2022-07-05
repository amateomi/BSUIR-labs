#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <unistd.h>
#include <fcntl.h>

const char port0[] = "/dev/tnt0";
const char port1[] = "/dev/tnt1";

#define SIZE 15
const char message[SIZE] = "Linux in cool!";
char       read_message[SIZE];

int main() {
  int fd0 = open(port0, O_RDWR | O_NOCTTY | O_NDELAY);
  if (fd0 < 0) {
    perror("port0 open");
    exit(errno);
  }
  int fd1 = open(port1, O_RDWR | O_NOCTTY | O_NDELAY);
  if (fd1 < 0) {
    perror("port1 open");
    exit(errno);
  }

  if (write(fd0, message, strlen(message)) < 0) {
    perror("write");
    exit(errno);
  }
  printf("message \"%s\" was written to %s\n", message, port0);
  close(fd0);

  if (read(fd1, read_message, strlen(message)) < 0) {
    perror("read");
    exit(errno);
  }
  if (strcmp(message, read_message) != 0) {
    fprintf(stderr, "read \"%s\" not equal to original message \"%s\"\n",
            read_message, message);
    exit(EXIT_FAILURE);
  }
  printf("read message \"%s\" from %s\n", read_message, port1);
  close(fd1);

  exit(EXIT_SUCCESS);
}