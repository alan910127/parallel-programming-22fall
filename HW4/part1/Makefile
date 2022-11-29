MPICXX := mpicxx
MPIFLAGS := -O3

MPIRUN := mpirun
RUNARGS := -np 8 -npernode 1 --hostfile ../hosts.txt

ifeq (run, $(firstword $(MAKECMDGOALS)))
	PROGRAM := $(wordlist 2, $(words $(MAKECMDGOALS)), $(MAKECMDGOALS))
endif

.PHONY: run clean

run: ${PROGRAM}
	${MPIRUN} ${RUNARGS} ${PROGRAM}

mpi_hello: hello.cc
	${MPICXX} ${MPIFLAGS} $^ -o $@
	parallel-scp -h ../hosts.txt $@ ~

clean:
	-rm mpi_hello
