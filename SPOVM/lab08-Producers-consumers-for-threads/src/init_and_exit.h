#ifndef INIT_AND_EXIT_H_
#define INIT_AND_EXIT_H_

#include <unistd.h>

#include "ipc.h"

void init(void);

void atexit_handler(void);

#endif // INIT_AND_EXIT_H_
