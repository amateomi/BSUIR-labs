#ifndef PROTOCOL_H_
#define PROTOCOL_H_

enum parent {
  PARENT_KILL,
  PARENT_FORCE_PRINT,
  PARENT_RESPONSE,
};

enum child {
  CHILD_ASK,
  CHILD_INFORM,
};

#endif //PROTOCOL_H_
