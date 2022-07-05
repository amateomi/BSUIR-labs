#include "producer.h"

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

extern msg_queue       queue;
extern pthread_mutex_t mutex;

extern sem_t* free_space;
extern sem_t* items;

extern pthread_t producers[];
extern size_t    producers_amount;

void create_producer(void) {
  if (producers_amount == CHILD_MAX - 1) {
    fputs("Max value of producers\n", stderr);
    return;
  }

  int res = pthread_create(&producers[producers_amount], NULL,
                           produce_handler, NULL);
  if (res) {
    fputs("Failed to create producer\n", stderr);
    exit(res);
  }

  ++producers_amount;
}

void remove_producer(void) {
  if (producers_amount == 0) {
    fputs("No producers to delete\n", stderr);
    return;
  }

  --producers_amount;
  pthread_cancel(producers[producers_amount]);
  pthread_join(producers[producers_amount], NULL);
}

_Noreturn void* produce_handler(void* arg) {
  msg_t  msg;
  size_t add_count_local;
  while (1) {
    produce_msg(&msg);

    sem_wait(free_space);

    pthread_mutex_lock(&mutex);
    add_count_local = put_msg(&msg);
    pthread_mutex_unlock(&mutex);

    sem_post(items);

    printf("%ld produce msg: hash=%X, add_count=%zu\n",
           pthread_self(), msg.hash, add_count_local);

    sleep(5);
  }
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
