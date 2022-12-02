MPICXX := mpicxx
MPIFLAGS := -O3

SRCS := $(wildcard *.cc)
OBJS := $(patsubst %.cc, %.o, ${SRCS})
TARGET := matmul

.PHONY: all push clean

all: ${TARGET}

push: ${TARGET}
	parallel-scp -h ../hosts.txt ${TARGET} ~

${TARGET}: ${OBJS}
	${MPICXX} ${MPIFLAGS} $^ -o $@

%.o: %.cc
	${MPICXX} ${MPIFLAGS} $^ -c -o $@

clean:
	-rm ${OBJS}
	-rm ${TARGET}