#ifndef DIRWALK__PATHS_H_
#define DIRWALK__PATHS_H_

#include <stddef.h>

#include "args.h"

typedef struct {
  unsigned char type;
  char          path[512]; // 256 defined by POSIX
} file_t;

typedef struct {
  size_t amount;
  file_t* data;
} files_t;

bool is_file_fits_flags(const file_t* file, flag_t flags);

void get_files(files_t* files, char* cur_path, flag_t flags);
void print(files_t files, flag_t flags);

#endif //DIRWALK__PATHS_H_
