#include <stdio.h>
#include <stdlib.h>

#include <netdb.h>

#include "helper.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: server <port>\n");
    exit(EXIT_FAILURE);
  }

  struct addrinfo* list = get_addrinfo_list(argv[1]);

  for (struct addrinfo* node = list; node != NULL; node = node->ai_next) {
    int server_socket = init_server(node);
    if (server_socket > 0) {
      serve(server_socket);
      exit(EXIT_SUCCESS);
    }
  }
  fprintf(stderr, "Failed to initialize server on service [%s]\n", argv[1]);
  exit(EXIT_FAILURE);
}
