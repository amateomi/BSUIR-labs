#ifndef PARENT__HELPER_H_
#define PARENT__HELPER_H_

void print_envp(char* envp[]);
char** create_child_env(char* fenvp);

char* search_child_path(char** str_arr);

#endif //PARENT__HELPER_H_
