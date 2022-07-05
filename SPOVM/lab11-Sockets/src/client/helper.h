#ifndef CLIENT_SRC_CLIENT_HELPER_H_
#define CLIENT_SRC_CLIENT_HELPER_H_

#include <sys/socket.h>

struct addrinfo* get_addrinfo_list(const char* hostname, const char* port);

int connect_to_server(const struct addrinfo* ai);

void communicate(int server_socket);

void* get_address(const struct sockaddr* sa);

#endif //CLIENT_SRC_CLIENT_HELPER_H_
