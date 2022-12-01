MPICXX := mpicxx
MPIFLAGS := -O3

MPIRUN := mpirun
RUNARGS := --hostfile ../hosts.txt

ifndef np
	RUNARGS += -np 8
else 
	RUNARGS += -np ${np}
endif

ifdef npernode
	RUNARGS += -npernode ${npernode}
endif

SRCS := $(wildcard *.cc)
OBJS := $(patsubst %.cc, %.o, ${SRCS})
TARGET := matmul

ifeq (run, $(firstword $(MAKECMDGOALS)))
	PROGRAM_ARGS := $(wordlist 2, $(words $(MAKECMDGOALS)), $(MAKECMDGOALS))
endif

.PHONY: all run clean

all: ${TARGET}

run: ${TARGET}
	parallel-scp -h ../hosts.txt ${TARGET} ~
	${MPIRUN} ${RUNARGS} ${TARGET} ${PROGRAM_ARGS}

${TARGET}: ${OBJS}
	${MPICXX} ${MPIFLAGS} $^ -o $@

%.o: %.cc
	${MPICXX} ${MPIFLAGS} $^ -c -o $@

clean:
	-rm ${OBJS}
	-rm ${TARGET}