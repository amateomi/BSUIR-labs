#include "helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>

#include <unistd.h>

#include "../protocol.h"

extern binary_t data;
extern stat_t   stat;

extern int alarm_count;

extern bool output_allowed;

void init(struct itimerval* timer) {
  // Init atexit
  if (atexit(atexit_handler)) {
    fputs("atexit failed", stderr);
    exit(EXIT_FAILURE);
  }

  // Init parent signals handler
  struct sigaction parent_signals;
  parent_signals.sa_flags     = SA_SIGINFO;
  parent_signals.sa_sigaction = parent_handler;

  if (sigaction(SIGUSR1, &parent_signals, NULL)) {
    perror("sigaction");
    exit(errno);
  }

  // Init alarm handler
  if (signal(SIGALRM, alarm_handler) == SIG_ERR) {
    perror("signal");
    exit(errno);
  }

  // Init timer
  timer->it_value.tv_sec     = 0;
  timer->it_value.tv_usec    = ALARM_DELAY;
  timer->it_interval.tv_sec  = 0;
  timer->it_interval.tv_usec = ALARM_DELAY;

  if (setitimer(ITIMER_REAL, timer, NULL)) {
    perror("setitimer");
    exit(errno);
  }
}

void atexit_handler(void) {
  printf("Process with pid=%d is ended\n", getpid());
}

void parent_handler(int sig, siginfo_t* info, void* context) {
  switch (info->si_int) {
    case PARENT_RESPONSE:
      output_allowed = true;
      break;

    case PARENT_FORCE_PRINT:
      print_stat();
      break;

    case PARENT_KILL:
      exit(EXIT_SUCCESS);

    default:
      fprintf(stderr,
              "Unknown signal occurred in parent_handler: %d\n",
              info->si_int);
      exit(EXIT_FAILURE);
  }
}

void alarm_handler(int sig) {
  data.I == 0 ? (data.O == 0 ? ++stat.O_O
                             : ++stat.O_I)
              : (data.O == 0) ? ++stat.I_O
                              : ++stat.I_I;

  ++alarm_count;
}

void ask_to_print(void) {
  union sigval value = {CHILD_ASK};
  if (sigqueue(getppid(), SIGUSR2, value)) {
    perror("sigqueue");
    exit(errno);
  }
}

void inform_about_print(void) {
  union sigval value = {CHILD_INFORM};
  if (sigqueue(getppid(), SIGUSR2, value)) {
    perror("sigqueue");
    exit(errno);
  }
}

void print_stat(void) {
  // Output is carried out character by character
  char buffer[1024];
  char* ptr = buffer;
  sprintf(buffer, "ppid=%d, pid=%d, stat={%zu, %zu, %zu, %zu}\n",
          getppid(), getpid(), stat.O_O, stat.O_I, stat.I_O, stat.I_I);
  while (*ptr) {
    putchar(*ptr);
    ++ptr;
  }
}

void reset_cycle(void) {
  alarm_count    = 0;
  output_allowed = false;
  stat.O_O = 0;
  stat.O_I = 0;
  stat.I_O = 0;
  stat.I_I = 0;
}
