#include "args.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <getopt.h>

char* parse_path(int argc, char* argv[]) {
  char* dir_to_walk = malloc(256);
  (argc == 1 || argv[1][0] == '-') ? strcpy(dir_to_walk, ".")
                                   : strcpy(dir_to_walk, argv[1]);

  size_t last_char = strlen(dir_to_walk) - 1;
  if (dir_to_walk[last_char] == '/') {
    dir_to_walk[last_char] = '\0';
  }

  return dir_to_walk;
}

flag_t parse_flags(int argc, char* argv[]) {
  flag_t flags          = {false, false, false, false};
  int    flag;
  bool   was_file_flags = false;
  while ((flag = getopt(argc, argv, "ldfs")) != -1) {
    if (flag == 'l' || flag == 'd' || flag == 'f') {
      was_file_flags = true;
    }
    switch (flag) {
      case 'l':
        flags.link = true;
        break;

      case 'd':
        flags.directory = true;
        break;

      case 'f':
        flags.regular_file = true;
        break;

      case 's':
        flags.sort = true;
        break;

      default:
        fprintf(stderr, "%c is invalid flag\n", flag);
        exit(EXIT_FAILURE);
    }
  }

  if (!was_file_flags) {
    flags.link = flags.directory = flags.regular_file = true;
  }

  return flags;
}