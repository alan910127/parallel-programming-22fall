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
EXECUTABLES := $(patsubst %.cc, %, $(SRCS))

ifeq (run, $(firstword $(MAKECMDGOALS)))
	PROGRAM := $(word 2, $(MAKECMDGOALS))
	PROGRAM_ARGS := $(wordlist 3, $(words $(MAKECMDGOALS)), $(MAKECMDGOALS))
endif

.PHONY: run clean

run: ${PROGRAM}
	parallel-scp -h ../hosts.txt ${PROGRAM} ~
	${MPIRUN} ${RUNARGS} ${PROGRAM} ${PROGRAM_ARGS}

%: %.cc
	${MPICXX} ${MPIFLAGS} $^ -o $@

clean:
	-rm ${EXECUTABLES}
