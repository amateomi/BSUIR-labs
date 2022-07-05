#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>

#include "helper.h"

size_t  amount = 0;
child_t child_array[MAX_CHILD];

int main(void) {
  init();

  printf("[+] - create child process\n"
         "[-] - kill last child process\n"
         "[k] - kill add child processes\n"
         "[s] <num> - stop statistic\n"
         "[g] <num> - resume statistic\n"
         "[p] num - print C_{num} statistic and freeze other childs\n"
         "[q] - exit\n"
         ">");
  while (1) {
    switch (getchar()) {
      case '+':
        create_child();
        printf("Child with pid=%d was created\n", child_array[amount - 1].pid);
        break;

      case '-':
        kill_last_child();
        printf("Last child was killed, %zu left\n", amount);
        break;

      case 'k':
        kill_all();
        puts("All child process was killed");
        break;

      case 's':
        call_corresponding_proc(stop_stat, stop_all_stat);
        break;

      case 'g':
        call_corresponding_proc(resume_stat, resume_all_stat);
        break;

      case 'p':
        if (!is_number_in_stdin()) {
          fputs("Invalid command! Correct form is p<num>", stderr);
          exit(EXIT_FAILURE);
        }
        size_t num = read_size_t();

        stop_all_stat();
        print_stat(child_array[num]);
        alarm(P_NUM_SEC_TIMER);
        break;

      case 'q':
        exit(EXIT_SUCCESS);

      default:
        break;
    }
  }
}
