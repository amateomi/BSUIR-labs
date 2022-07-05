#include "init_and_exit.h"

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <errno.h>

#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <unistd.h>
#include <fcntl.h>

#define OPEN_FLAGS (O_RDWR | O_CREAT | O_TRUNC)
#define MODE (S_IRUSR | S_IWUSR)

extern msg_queue* queue;
extern sem_t    * mutex;

extern sem_t* free_space;
extern sem_t* items;

extern pid_t  producers[];
extern size_t producers_amount;

extern pid_t  consumers[];
extern size_t consumers_amount;

static pid_t parent_pid;

void init(void) {
  parent_pid = getpid();

  atexit(atexit_handler);

  // Setup shared memory
  int fd = shm_open(SHM_OBJECT, OPEN_FLAGS, MODE);
  if (fd < 0) {
    perror("shm_open");
    exit(errno);
  }

  if (ftruncate(fd, sizeof(msg_queue))) {
    perror("ftruncate");
    exit(errno);
  }

  void* ptr = mmap(NULL, sizeof(msg_queue), PROT_READ | PROT_WRITE,
                   MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("mmap");
    exit(errno);
  }

  queue = (msg_queue*) ptr;

  if (close(fd)) {
    perror("close");
    exit(errno);
  }

  // Setup queue
  msg_queue_init();

  // Setup semaphores
  if ((mutex = sem_open(MUTEX, OPEN_FLAGS, MODE, 1)) == SEM_FAILED ||
      (free_space = sem_open(FREE_SPACE, OPEN_FLAGS, MODE, MSG_MAX))
          == SEM_FAILED ||
      (items = sem_open(ITEMS, OPEN_FLAGS, MODE, 0)) == SEM_FAILED) {
    perror("sem_open");
    exit(errno);
  }
}

void atexit_handler(void) {
  if (getpid() == parent_pid) {
    for (size_t i = 0; i < producers_amount; ++i) {
      kill(producers[i], SIGKILL);
      wait(NULL);
    }
    for (size_t i = 0; i < consumers_amount; ++i) {
      kill(consumers[i], SIGKILL);
      wait(NULL);
    }
  } else if (getppid() == parent_pid) {
    kill(getppid(), SIGKILL);
  }

  if (shm_unlink(SHM_OBJECT)) {
    perror("shm_unlink");
    abort();
  }
  if (sem_unlink(MUTEX) ||
      sem_unlink(FREE_SPACE) ||
      sem_unlink(ITEMS)) {
    perror("sem_unlink");
    abort();
  }
}
