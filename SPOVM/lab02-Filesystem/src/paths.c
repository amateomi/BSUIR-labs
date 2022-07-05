#include "paths.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <dirent.h>

bool is_file_fits_flags(const file_t* file, flag_t flags) {
  switch (file->type) {
    case DT_LNK:
      return flags.link;

    case DT_DIR:
      return flags.directory;

    case DT_REG:
      return flags.regular_file;

    default:
      return true;
  }
}

void get_files(files_t* files, char* cur_path, flag_t flags) {
  strcat(cur_path, "/");
  DIR* dir = opendir(cur_path);

  if (!dir) {
    if (errno == EACCES) {
      perror(cur_path);
      return;
    } else {
      perror(cur_path);
      exit(errno);
    }
  }

  struct dirent* file_in_dir;
  while ((file_in_dir = readdir(dir))) {
    if (!strcmp(file_in_dir->d_name, ".") ||
        !strcmp(file_in_dir->d_name, "..")) {
      continue;
    }

    file_t file = {file_in_dir->d_type, {'\0'}};
    strcpy(file.path, cur_path);
    strcat(file.path, file_in_dir->d_name);

    if (is_file_fits_flags(&file, flags)) {
      files->data[files->amount++] = file;
    }

    if (file_in_dir->d_type == DT_DIR) {
      get_files(files, file.path, flags);
    }
  }

  closedir(dir);
}

static int cmpfile(const void* f1, const void* f2) {
  return strcmp(((const file_t*) f1)->path,
                ((const file_t*) f2)->path);
}

void print(files_t files, flag_t flags) {
  if (flags.sort) {
    qsort(files.data, files.amount, sizeof(file_t), cmpfile);
  }
  for (size_t i = 0; i < files.amount; ++i) {
    switch (files.data[i].type) {
      case DT_LNK: // green
        printf("\033[1;36m%s\033[0m\n", files.data[i].path);
        break;

      case DT_DIR: // blue
        printf("\033[1;34m%s\033[0m\n", files.data[i].path);
        break;

      case DT_REG: // uncolored
        puts(files.data[i].path);
        break;

      default: // purple
        printf("\033[1;35m%s\033[0m\n", files.data[i].path);
        break;
    }
  }
}
