default: conv

.PHONY: cudaconv

CC = gcc-10
FLAGS = -O3 -lOpenCL -m64 -ffloat-store -w -g

OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o
CUDAOBJS = $(addprefix cudaobj/, $(OBJS))

conv: $(OBJS)
	$(CC) -o $@ $(OBJS) $(FLAGS)

cudaconv: 
	make -f Makefile.cuda

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@

clean:
	rm -f conv *.o output.bmp ref.bmp