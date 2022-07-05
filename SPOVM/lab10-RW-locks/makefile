CC = g++
CFLAGS = -O2 -std=c++17 -pedantic -Wall -Wextra -Werror
CFLAGS += -pthread

_SRC = app.cpp code_index.cpp deletion_accounting.cpp \
       directory_notify_manager.cpp file_manager.cpp main.cpp master_index.cpp \
       name_index.cpp record.cpp rwlock_manager.cpp table.cpp
SRC = $(patsubst %,src/%,$(_SRC))

database: $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: run
run:
	make database
	./database product/table
	make clean

.PHONY: clean
clean:
	rm database
