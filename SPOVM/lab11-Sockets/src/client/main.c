#include <stdio.h>
#include <stdlib.h>

#include <netdb.h>

#include "helper.h"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: client <hostname> <port>\n");
    exit(EXIT_FAILURE);
  }

  struct addrinfo* list = get_addrinfo_list(argv[1], argv[2]);

  for (struct addrinfo* node = list; node != NULL; node = node->ai_next) {
    int server_socket = connect_to_server(node);
    if (server_socket > 0) {
      communicate(server_socket);
      exit(EXIT_SUCCESS);
    }
  }
  fprintf(stderr, "Connection to hostname [%s] is failed\n", argv[1]);
  exit(EXIT_FAILURE);
}
