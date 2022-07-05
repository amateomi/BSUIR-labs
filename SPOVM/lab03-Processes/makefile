CC = gcc
CFLAGS = -O2 -std=c11 -pedantic -Wall -Wextra -Werror -Wno-unused-parameter

CHILD = src/child/main.c

_PARENT = main.c helper.c
PARENT = $(patsubst %,src/parent/%,$(_PARENT))

child: $(CHILD)
	$(CC) $(CFLAGS) -o $@ $^

parent: $(PARENT)
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: run
run:
	make child
	make parent
	CHILD_PATH=./child ./parent "env.txt"
	make clean

.PHONY: clean
clean:
	rm child parent
