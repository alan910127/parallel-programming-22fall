#!/bin/sh

VERSION="${1:-1}"
VIEW="$2"
MAX_ITERATION="$3"

cd /home/109652039/HW5/src

cp "kernel${VERSION}.cu" kernel.cu
make
./mandelbrot --gpu-only 1 --view "${VIEW}" --iter "${MAX_ITERATION}"

cd -