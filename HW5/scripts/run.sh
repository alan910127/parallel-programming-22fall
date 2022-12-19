#!/bin/sh

VERSION="${1:-1}"

cd /home/109652039/HW5/src

cp "kernel${VERSION}.cu" kernel.cu
make
./mandelbrot

cd -