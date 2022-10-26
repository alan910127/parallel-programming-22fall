#include "myrandom.h"

// source: https://codeforces.com/blog/entry/61587
int fast_rand(rand_state_t* state) {
    return (((*state = *state * 214013L + 2531011L) >> 16) & 0x7fff);
}

double get_random(rand_state_t* state) {
    return (double)fast_rand(state) / 0x7fff;
}