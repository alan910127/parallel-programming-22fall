CC := gcc
CFLAGS := -Ofast -Wall -lpthread

ifeq ($(CHECK), 1)
	CFLAGS += -DCHECK_ARGS
endif

TARGET := pi.out
SOURCE := $(wildcard *.c)
OBJECT := $(patsubst %.c,%.o,${SOURCE})

.PHONY: all clean

all: ${TARGET}

${TARGET}: ${OBJECT}
	${CC} ${CFLAGS} $^ -o $@

%.o: %.c
	${CC} ${CFLAGS} $^ -c -o $@

clean:
	-rm ${TARGET}
	-rm ${OBJECT}

