#include <stdbool.h>

#include "helper.h"

static const binary_t zero  = {0, 0};
static const binary_t three = {1, 1};

binary_t data;
stat_t   stat;

int alarm_count;

bool output_allowed;

int main(void) {
  struct itimerval timer;
  init(&timer);

  while (1) {
    data = zero;
    data = three;

    if (alarm_count == ALARM_COUNTS_TO_PRINT) {
      ask_to_print();
      if (output_allowed) {
        print_stat();
        inform_about_print();
      }
      reset_cycle();
    }
  }
}
