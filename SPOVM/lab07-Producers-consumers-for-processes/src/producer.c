#include "producer.h"

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <errno.h>

#include <sys/wait.h>
#include <semaphore.h>
#include <unistd.h>

extern msg_queue* queue;
extern sem_t    * mutex;

extern sem_t* free_space;
extern sem_t* items;

extern pid_t  producers[];
extern size_t producers_amount;

void create_producer(void) {
  if (producers_amount == CHILD_MAX - 1) {
    fputs("Max value of producers\n", stderr);
    return;
  }

  switch (producers[producers_amount] = fork()) {
    default:
      // Parent process
      ++producers_amount;
      return;

    case 0:
      // Child process
      srand(getpid());
      break;

    case -1:
      perror("fork");
      exit(errno);
  }

  msg_t  msg;
  size_t add_count_local;
  while (1) {
    produce_msg(&msg);

    sem_wait(free_space);

    sem_wait(mutex);
    add_count_local = put_msg(&msg);
    sem_post(mutex);

    sem_post(items);

    printf("%d produce msg: hash=%X, add_count=%zu\n",
           getpid(), msg.hash, add_count_local);

    sleep(5);
  }

}

void remove_producer(void) {
  if (producers_amount == 0) {
    fputs("No producers to delete\n", stderr);
    return;
  }

  --producers_amount;
  kill(producers[producers_amount], SIGKILL);
  wait(NULL);
}

void produce_msg(msg_t* msg) {
  size_t value = rand() % 257;
  if (value == 256) {
    msg->type = -1;
  } else {
    msg->type = 0;
    msg->size = (uint8_t) value;
  }

  for (size_t i = 0; i < value; ++i) {
    msg->data[i] = (char) (rand() % 256);
  }

  msg->hash = 0;
  msg->hash = hash(msg);
}
