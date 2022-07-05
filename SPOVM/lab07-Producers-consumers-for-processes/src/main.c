#include <stdio.h>
#include <stdlib.h>

#include <semaphore.h>

#include "init_and_exit.h"
#include "producer.h"
#include "consumer.h"

static const char options[] = {"Options:\n"
                               "[o] - print options\n"
                               "[P] - create producer\n"
                               "[p] - delete producer\n"
                               "[C] - create consumer\n"
                               "[c] - delete consumer\n"
                               "[q] - quit"};

msg_queue* queue;
sem_t    * mutex;

sem_t* free_space;
sem_t* items;

pid_t  producers[CHILD_MAX];
size_t producers_amount;

pid_t  consumers[CHILD_MAX];
size_t consumers_amount;

int main(void) {
  init();

  puts(options);
  while (1) {
    switch (getchar()) {
      case 'o':
        puts(options);
        break;

      case 'P':
        create_producer();
        break;

      case 'p':
        remove_producer();
        break;

      case 'C':
        create_consumer();
        break;

      case 'c':
        remove_consumer();
        break;

      case 'q':
        exit(EXIT_SUCCESS);

      case EOF:
        fputs("Input error", stderr);
        exit(EXIT_FAILURE);

      default:
        break;
    }
  }

}
