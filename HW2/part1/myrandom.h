#ifndef HW2_PART1_MYRANDOM_H
#define HW2_PART1_MYRANDOM_H

#include <stdint.h>

typedef long rand_state_t;

int fast_rand(rand_state_t* state);
double get_random(rand_state_t* state);

#endif