#include "helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <ctype.h>

#include <sys/wait.h>
#include <unistd.h>

extern size_t  amount;
extern child_t child_array[MAX_CHILD];

// Single child actions

void create_child(void) {
  if (amount == MAX_CHILD - 1) {
    fputs("Child amount overflow", stderr);
    exit(EXIT_FAILURE);
  }

  pid_t ret = fork();
  switch (ret) {
    case 0: {
      // Child process
      char child_name[8];
      sprintf(child_name, "C_%zu", amount);
      if (execl("./child", child_name, NULL) == -1) {
        perror("execl");
        exit(errno);
      }
    }
      break;

    case -1:
      perror("fork");
      exit(errno);

    default:
      // Parent process
      child_array[amount].pid           = ret;
      child_array[amount].print_allowed = true;
      ++amount;
      break;
  }
}

pid_t kill_last_child(void) {
  if (amount == 0) {
    fputs("child_array is empty", stderr);
    exit(EXIT_FAILURE);
  }
  send_signal(child_array[--amount], PARENT_KILL);
  return wait(NULL);
}

void stop_stat(child_t* child) {
  child->print_allowed = false;
}

void resume_stat(child_t* child) {
  child->print_allowed = true;
}

void print_stat(child_t child) {
  send_signal(child, PARENT_FORCE_PRINT);
}

// All child actions

void kill_all(void) {
  while (amount > 0) {
    kill_last_child();
  }
}

void stop_all_stat(void) {
  for (size_t i = 0; i < amount; ++i) {
    stop_stat(&child_array[i]);
  }
}

void resume_all_stat(void) {
  for (size_t i = 0; i < amount; ++i) {
    resume_stat(&child_array[i]);
  }
}

// Utility

void init(void) {
  // Init atexit
  if (atexit(atexit_handler)) {
    fputs("atexit failed", stderr);
    exit(EXIT_FAILURE);
  }

  // Init alarm handler
  if (signal(SIGALRM, alarm_handler) == SIG_ERR) {
    perror("signal");
    exit(errno);
  }

  // Init child signals handler
  struct sigaction child_signals;
  child_signals.sa_flags     = SA_SIGINFO;
  child_signals.sa_sigaction = child_handler;

  if (sigaction(SIGUSR2, &child_signals, NULL)) {
    perror("sigaction");
    exit(errno);
  }
}

void atexit_handler(void) {
  kill_all();
  puts("All child processes are killed, program is finished");
}

void child_handler(int sig, siginfo_t* info, void* context) {
  size_t i = get_index_by_pid(info->si_pid);

  switch (info->si_int) {
    case CHILD_ASK:
      if (child_array[i].print_allowed) {
        send_signal(child_array[i], PARENT_RESPONSE);
      }
      break;

    case CHILD_INFORM:
      printf("%d informed parent\n", child_array[i].pid);
      break;

    default:
      fprintf(stderr,
              "Unknown signal occurred in child_handler: %d\n",
              info->si_int);
      exit(EXIT_FAILURE);
  }
}

void alarm_handler(int sig) {
  resume_all_stat();
}

void send_signal(child_t child, enum parent signal) {
  union sigval value = {signal};
  if (sigqueue(child.pid, SIGUSR1, value)) {
    perror("sigqueue");
    exit(errno);
  }
}

size_t get_index_by_pid(pid_t pid) {
  for (size_t i = 0; i < amount; ++i) {
    if (child_array[i].pid == pid) {
      return i;
    }
  }
  fprintf(stderr, "pid=%d is not child of the process\n", pid);
  exit(EXIT_FAILURE);
}

void call_corresponding_proc(void (* single)(child_t*),
                             void (* all)(void)) {
  if (is_number_in_stdin()) {
    size_t num = read_size_t();
    if (validate_num(num)) {
      single(&child_array[num]);
    }
  } else {
    all();
  }
}

bool validate_num(size_t num) {
  if (num >= amount) {
    fputs("Num is greater than child_array amount", stderr);
    return false;
  }
  return true;
}

bool is_number_in_stdin(void) {
  int  c         = getc(stdin);
  bool is_number = isdigit(c);
  ungetc(c, stdin);
  return is_number;
}

size_t read_size_t(void) {
  size_t num;
  if (scanf("%zu", &num) != 1) {
    fputs("Scanf read error", stderr);
    exit(EXIT_FAILURE);
  }
  return num;
}
