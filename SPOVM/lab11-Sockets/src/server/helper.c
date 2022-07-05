#include "helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <sys/wait.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

#define HOST_NAME_LEN 1024
#define CONNECTION_QUEUE 10

struct addrinfo* get_addrinfo_list(const char* port) {
  char hostname[HOST_NAME_LEN];
  if (gethostname(hostname, sizeof(hostname)) < 0) {
    perror("gethostname");
    exit(errno);
  }

  struct addrinfo hints;
  memset(&hints, 0, sizeof hints);
  hints.ai_family   = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  struct addrinfo* result = NULL;

  int error = getaddrinfo(hostname, port, &hints, &result);
  if (error) {
    fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(error));
    exit(error);
  }

  return result;
}

int init_server(const struct addrinfo* ai) {
  int server_socket = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
  if (server_socket < 0) {
    perror("socket");
    return -1;
  }

  const int yes = 1;
  // Helps with "Address already in use" error
  if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes))) {
    perror("setsockopt");
    return -1;
  }

  if (bind(server_socket, ai->ai_addr, ai->ai_addrlen)) {
    perror("bind");
    close(server_socket);
    return -1;
  }

  if (listen(server_socket, CONNECTION_QUEUE)) {
    perror("listen");
    close(server_socket);
    return -1;
  }

  char ip_buffer[INET6_ADDRSTRLEN];
  if (!inet_ntop(ai->ai_family, get_address(ai->ai_addr),
                 ip_buffer, sizeof(ip_buffer))) {
    perror("inet_ntop");
    return -1;
  }
  printf("Server initialized on IP [%s]\n", ip_buffer);

  return server_socket;
}

void serve(int server_socket) {
  puts("Waiting for connections...");

  struct sockaddr_storage client;

  socklen_t len = sizeof(client);
  char      ip_buffer[INET6_ADDRSTRLEN];
  while (1) {
    int client_socket = accept(server_socket, (struct sockaddr*) &client, &len);
    if (client_socket < 0) {
      perror("accept");
      continue;
    }

    inet_ntop(client.ss_family, get_address((struct sockaddr*) &client),
              ip_buffer, sizeof(ip_buffer));
    printf("%s connected to server\n", ip_buffer);

    pid_t pid = fork();
    if (pid < 0) {
      perror("fork");

    } else if (pid) { // Parent process
      close(client_socket);
      wait(NULL);

    } else { // Child process
      if (dup2(client_socket, STDIN_FILENO) != STDIN_FILENO ||
          dup2(client_socket, STDOUT_FILENO) != STDOUT_FILENO ||
          dup2(client_socket, STDERR_FILENO) != STDERR_FILENO) {
        perror("dup2");
        exit(errno);
      }
      close(client_socket);

      execl("../lab10-RW-locks/database",
            "database", "../lab10-RW-locks/product/table", NULL);
      perror("execl");
    }
  }
}

void* get_address(const struct sockaddr* sa) {
  if (sa->sa_family == AF_INET) {
    return &(((struct sockaddr_in*) sa)->sin_addr);
  }
  return &(((struct sockaddr_in6*) sa)->sin6_addr);
}
