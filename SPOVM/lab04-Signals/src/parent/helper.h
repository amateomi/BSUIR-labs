#ifndef PARENT__HELPER_H_
#define PARENT__HELPER_H_

#include <stdbool.h>
#include <signal.h>

#include <sys/types.h>

#include "../protocol.h"

#define MAX_CHILD 1024
#define P_NUM_SEC_TIMER 5

typedef struct {
  pid_t pid;
  bool  print_allowed;
} child_t;

// Single child actions

void create_child(void);
pid_t kill_last_child(void);
void stop_stat(child_t* child);
void resume_stat(child_t* child);
void print_stat(child_t child);

// All child actions

void kill_all(void);
void stop_all_stat(void);
void resume_all_stat(void);

// Utility

void init(void);

void atexit_handler(void);
void child_handler(int sig, siginfo_t* info, void* context);
void alarm_handler(int sig);

void send_signal(child_t child, enum parent signal);

size_t get_index_by_pid(pid_t pid);

void call_corresponding_proc(void (* single)(child_t*),
                             void (* all)(void));
bool validate_num(size_t num);
bool is_number_in_stdin(void);
size_t read_size_t(void);

#endif //PARENT__HELPER_H_
