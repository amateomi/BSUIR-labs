#include "consumer.h"

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

extern msg_queue       queue;
extern pthread_mutex_t mutex;

extern sem_t* free_space;
extern sem_t* items;

extern pthread_t consumers[];
extern size_t    consumers_amount;

void create_consumer(void) {
  if (consumers_amount == CHILD_MAX - 1) {
    fputs("Max value of consumers\n", stderr);
    return;
  }

  int res = pthread_create(&consumers[consumers_amount], NULL,
                           consume_handler, NULL);
  if (res) {
    fputs("Failed to create producer\n", stderr);
    exit(res);
  }

  ++consumers_amount;
}

void remove_consumer(void) {
  if (consumers_amount == 0) {
    fputs("No consumers to delete\n", stderr);
    return;
  }

  --consumers_amount;
  pthread_cancel(consumers[consumers_amount]);
  pthread_join(consumers[consumers_amount], NULL);
}

_Noreturn void* consume_handler(void* arg) {
  msg_t  msg;
  size_t extract_count_local;
  while (1) {
    sem_wait(items);

    pthread_mutex_lock(&mutex);
    extract_count_local = get_msg(&msg);
    pthread_mutex_unlock(&mutex);

    sem_post(free_space);

    consume_msg(&msg);

    printf("%ld consume msg: hash=%X, extract_count=%zu\n",
           pthread_self(), msg.hash, extract_count_local);

    sleep(5);
  }
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
