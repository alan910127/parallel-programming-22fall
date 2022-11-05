#!/bin/sh

make -C part1 clean
make -C part2 clean

scp -r $(pwd) pp-machine:~