#include "consumer.h"

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

extern pid_t  consumers[];
extern size_t consumers_amount;

void create_consumer(void) {
  if (consumers_amount == CHILD_MAX - 1) {
    fputs("Max value of consumers\n", stderr);
    return;
  }

  switch (consumers[consumers_amount] = fork()) {
    default:
      // Parent process
      ++consumers_amount;
      return;

    case 0:
      // Child process
      break;

    case -1:
      perror("fork");
      exit(errno);
  }

  msg_t  msg;
  size_t extract_count_local;
  while (1) {
    sem_wait(items);

    sem_wait(mutex);
    extract_count_local = get_msg(&msg);
    sem_post(mutex);

    sem_post(free_space);

    consume_msg(&msg);

    printf("%d consume msg: hash=%X, extract_count=%zu\n",
           getpid(), msg.hash, extract_count_local);

    sleep(5);
  }

}

void remove_consumer(void) {
  if (consumers_amount == 0) {
    fputs("No consumers to delete\n", stderr);
    return;
  }

  --consumers_amount;
  kill(consumers[consumers_amount], SIGKILL);
  wait(NULL);
}

void consume_msg(msg_t* msg) {
  uint16_t msg_hash = msg->hash;
  msg->hash = 0;
  uint16_t check_sum = hash(msg);
  if (msg_hash != check_sum) {
    fprintf(stderr, "check_sum=%X not equal msg_hash=%X\n",
            check_sum, msg_hash);
  }
  msg->hash = msg_hash;
}
