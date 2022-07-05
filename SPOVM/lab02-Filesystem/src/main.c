#include <stdlib.h>

#include "args.h"
#include "paths.h"

int main(int argc, char* argv[]) {
  char* dir_to_walk = parse_path(argc, argv);
  flag_t flags = parse_flags(argc, argv);

  files_t files = {0, (file_t*) malloc(30000000 * sizeof(file_t))};

  get_files(&files, dir_to_walk, flags);
  print(files, flags);

  exit(EXIT_SUCCESS);
}
