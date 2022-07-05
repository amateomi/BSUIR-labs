#ifndef SERVER_SRC_SERVER_HELPER_H_
#define SERVER_SRC_SERVER_HELPER_H_

#include <sys/socket.h>

struct addrinfo* get_addrinfo_list(const char* port);

int init_server(const struct addrinfo* ai);

void serve(int server_socket);

void* get_address(const struct sockaddr* sa);

#endif //SERVER_SRC_SERVER_HELPER_H_
