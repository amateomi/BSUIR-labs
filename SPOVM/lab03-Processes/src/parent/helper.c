#include "helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

const char child_path[] = "CHILD_PATH";

static int qsort_cmp(const void* str1, const void* str2) {
  return strcmp(*(const char**) str1, *(const char**) str2);
}

void print_envp(char* envp[]) {
  size_t envpc = 0;
  while (envp[envpc]) {
    ++envpc;
  }

  qsort(envp, envpc, sizeof(char*), qsort_cmp);

  printf("\033[1;35m"); // Purple color

  puts("Parent environment variables:");
  for (size_t i = 0; i < envpc; ++i) {
    puts(envp[i]);
  }

  printf("\033[0m"); // Turn off purple color
}

char** create_child_env(char* fenvp) {
  FILE* stream = fopen(fenvp, "r");
  if (!stream) {
    perror("fopen");
    exit(errno);
  }

  char** env = malloc(sizeof(char*));
  size_t i = 0;

  const int MAX_SIZE = 256;
  char      buffer[MAX_SIZE];
  while (fgets(buffer, MAX_SIZE, stream) != NULL) {
    buffer[strcspn(buffer, "\n")] = '\0';

    char* env_val = getenv(buffer);
    if (env_val) {
      env[i] = malloc((strlen(buffer) + strlen(env_val) + 2) * sizeof(char));
      strcat(strcat(strcpy(env[i], buffer), "="), env_val);
      env = realloc(env, (++i + 1) * sizeof(char*));
    }
  }
  env[i] = NULL;

  return env;
}

char* search_child_path(char** str_arr) {
  while (*str_arr) {
    if (!strncmp(*str_arr, child_path, strlen(child_path))) {
      return *str_arr + strlen(child_path) + 1; // skip CHILD_PATH=
    }
    ++str_arr;
  }
  return NULL;
}