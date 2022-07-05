#ifndef DIRWALK__ARGS_H_
#define DIRWALK__ARGS_H_

#include <stdbool.h>

typedef struct {
  bool link;
  bool directory;
  bool regular_file;
  bool sort;
} flag_t;

char* parse_path(int argc, char* argv[]);
flag_t parse_flags(int argc, char* argv[]);

#endif //DIRWALK__ARGS_H_
