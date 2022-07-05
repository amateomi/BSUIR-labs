#include "helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

#define MAX_SLEEP 4
#define BUF_LEN 1024

struct addrinfo* get_addrinfo_list(const char* hostname, const char* port) {
  struct addrinfo hint;
  memset(&hint, 0, sizeof(hint));
  hint.ai_family   = AF_UNSPEC;
  hint.ai_socktype = SOCK_STREAM;

  struct addrinfo* result = NULL;

  int error = getaddrinfo(hostname, port, &hint, &result);
  if (error) {
    fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(error));
    exit(error);
  }

  return result;
}

int connect_to_server(const struct addrinfo* ai) {
  int server_socket;
  int sleep_time;
  for (sleep_time = 1; sleep_time <= MAX_SLEEP; sleep_time *= 2) {
    server_socket = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (server_socket < 0) {
      perror("socket");
      return -1;
    }

    if (!connect(server_socket, ai->ai_addr, ai->ai_addrlen)) {
      break;
    }
    perror("connect");
    sleep(sleep_time);
    close(server_socket);
  }
  if (sleep_time > MAX_SLEEP) {
    close(server_socket);
    return -1;
  }

  char ip_buffer[INET6_ADDRSTRLEN];
  if (!inet_ntop(ai->ai_family, get_address(ai->ai_addr),
                 ip_buffer, sizeof(ip_buffer))) {
    perror("inet_ntop");
    close(server_socket);
    return -1;
  }
  printf("Connected to %s\n", ip_buffer);

  return server_socket;
}

void communicate(int server_socket) {
  char    buffer[BUF_LEN];
  ssize_t n;
  while (1) {
    n = recv(server_socket, buffer, BUF_LEN, 0);
    if (n == 0) {
      exit(EXIT_SUCCESS);
    }
    if (n < 0) {
      perror("recv");
      exit(errno);
    }
    write(STDOUT_FILENO, buffer, n);

    if (!fgets(buffer, BUF_LEN - 1, stdin)) {
      fprintf(stderr, "Failed to read input\n");
      exit(EXIT_FAILURE);
    }
    n = send(server_socket, buffer, strlen(buffer), 0);
    if (n < 0) {
      perror("send");
      exit(errno);
    }
    memset(buffer, 0, BUF_LEN);
  }
}

void* get_address(const struct sockaddr* sa) {
  if (sa->sa_family == AF_INET) {
    return &(((struct sockaddr_in*) sa)->sin_addr);
  }
  return &(((struct sockaddr_in6*) sa)->sin6_addr);
}
