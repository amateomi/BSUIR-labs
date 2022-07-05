CC = gcc
CFLAGS = -O2 -std=c11 -pedantic -Wall -Wextra -Werror -Wno-unused-parameter -D_DEFAULT_SOURCE
CFLAGS += -pthread

_SRC = main.c consumer.c init_and_exit.c ipc.c producer.c
SRC = $(patsubst %,src/%,$(_SRC))

ipc: $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: run
run:
	make ipc
	./ipc
	make clean

clean:
	rm ipc
