CC = gcc
CFLAGS = -std=c11 -pedantic -Wall -Wextra -D_DEFAULT_SOURCE

_CLIENT = main.c helper.c
CLIENT = $(patsubst %,src/client/%,$(_CLIENT))

_SERVER = main.c helper.c
SERVER = $(patsubst %,src/server/%,$(_SERVER))

client: $(CLIENT)
	$(CC) $(CFLAGS) -o $@ $^

server: $(SERVER)
	$(CC) $(CFLAGS) -o $@ $^

# `make run` start server, client must be started manually
.PHONY: run
run:
	make -C ../lab10*
	make client
	make server
	./server 1337
	make clean

.PHONY: clean
clean:
	rm client
	rm server
