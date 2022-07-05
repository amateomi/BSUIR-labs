#include "init_and_exit.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <semaphore.h>
#include <fcntl.h>

#define OPEN_FLAGS (O_RDWR | O_CREAT | O_TRUNC)
#define MODE (S_IRUSR | S_IWUSR)

extern msg_queue       queue;
extern pthread_mutex_t mutex;

extern sem_t* free_space;
extern sem_t* items;

void init(void) {
  atexit(atexit_handler);

  // Setup queue
  msg_queue_init();

  // Setup mutex
  int res = pthread_mutex_init(&mutex, NULL);
  if (res) {
    fputs("Failed mutex init\n", stderr);
    exit(res);
  }

  // Setup semaphores
  if ((free_space = sem_open(FREE_SPACE, OPEN_FLAGS, MODE, MSG_MAX))
      == SEM_FAILED ||
      (items = sem_open(ITEMS, OPEN_FLAGS, MODE, 0)) == SEM_FAILED) {
    perror("sem_open");
    exit(errno);
  }
}

void atexit_handler(void) {
  int res = pthread_mutex_destroy(&mutex);
  if (res) {
    fputs("Failed mutex destroy\n", stderr);
    exit(res);
  }
  if (sem_unlink(FREE_SPACE) ||
      sem_unlink(ITEMS)) {
    perror("sem_unlink");
    abort();
  }
}
