#ifndef PRODUCER_H_
#define PRODUCER_H_

#include "ipc.h"

void create_producer(void);
void remove_producer(void);

void* produce_handler(void* arg);
void produce_msg(msg_t* msg);

#endif // PRODUCER_H_
