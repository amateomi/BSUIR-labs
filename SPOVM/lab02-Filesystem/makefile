CC = gcc
CFLAGS = -O2 -std=c11 -pedantic -Wall -Wextra -Werror -D_DEFAULT_SOURCE

_SRC = main.c args.c paths.c
SRC = $(patsubst %,src/%,$(_SRC))

dirwalk: $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: run
run:
	make dirwalk
	./dirwalk
	make clean

clean:
	rm dirwalk
