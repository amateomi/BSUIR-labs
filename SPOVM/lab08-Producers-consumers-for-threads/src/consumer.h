#ifndef CONSUMER_H_
#define CONSUMER_H_

#include "ipc.h"

void create_consumer(void);
void remove_consumer(void);

void* consume_handler(void* arg);
void consume_msg(msg_t* msg);

#endif // CONSUMER_H_
