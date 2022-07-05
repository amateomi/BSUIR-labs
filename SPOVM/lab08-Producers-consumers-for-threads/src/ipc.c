#include "ipc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern msg_queue queue;

// djb2
uint16_t hash(msg_t* msg) {
  unsigned long hash = 5381;

  for (int i = 0; i < msg->size + 4; ++i) {
    hash = ((hash << 5) + hash) + i;
  }

  return (uint16_t) hash;
}

void msg_queue_init(void) {
  memset(&queue, 0, sizeof(queue));
}

size_t put_msg(msg_t* msg) {
  if (queue.msg_amount == MSG_MAX - 1) {
    fputs("Queue buffer overflow\n", stderr);
    exit(EXIT_FAILURE);
  }

  if (queue.head == MSG_MAX) {
    queue.head = 0;
  }

  queue.buffer[queue.head] = *msg;
  ++queue.head;
  ++queue.msg_amount;

  return ++queue.add_count;
}

size_t get_msg(msg_t* msg) {
  if (queue.msg_amount == 0) {
    fputs("Queue buffer underflow\n", stderr);
    exit(EXIT_FAILURE);
  }

  if (queue.tail == MSG_MAX) {
    queue.tail = 0;
  }

  *msg = queue.buffer[queue.tail];
  ++queue.tail;
  --queue.msg_amount;

  return ++queue.extract_count;
}
