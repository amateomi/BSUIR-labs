CC = gcc
CFLAGS = -std=c11 -pedantic -Wall -Wextra -Werror -Wno-unused-parameter -D_DEFAULT_SOURCE

_CHILD = main.c helper.c
CHILD = $(patsubst %,src/child/%,$(_CHILD))

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
	./parent
	make clean

.PHONY: clean
clean:
	rm child parent
