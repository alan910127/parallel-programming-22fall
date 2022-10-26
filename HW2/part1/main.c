#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>

#include "myrandom.h"

typedef struct argument_s {
	long long num_tosses;
	rand_state_t rand_seed;
} argument_t;

void* calculate_pi(void* params) {
	argument_t* args = (argument_t*)params;
	long long num_tosses = args->num_tosses;
	rand_state_t rand_seed = args->rand_seed;
	long long num_hits = 0;

	for (long long i = 0; i < num_tosses; ++i) {
		double x = get_random(&rand_seed);
		double y = get_random(&rand_seed);
		if (x * x + y * y <= 1.0) {
			++num_hits;
		}
	}

	pthread_exit((void*)num_hits);
}

int main(int argc, char** argv) {
#ifdef CHECK_ARGS
	if (argc != 3) {
		fprintf(stderr, "usage: %s threads tosses\n", argv[0]);
		return -1;
	}

	char* invalid_char = NULL;
	int num_threads = strtol(argv[1], &invalid_char, 10);
	if (*invalid_char != '\0') {
		fprintf(stderr, "invalid input: threads (expected an integer, got '%s')\n", argv[1]);
		return -1;
	}

	invalid_char = NULL;
	long long num_tosses = strtoll(argv[2], &invalid_char, 10);
	if (*invalid_char != '\0') {
		fprintf(stderr, "invalid input: tosses (expected a 64-bit integer, got '%s')\n", argv[2]);
		return -1;
	}
#else
	int num_threads = strtol(argv[1], NULL, 10);
	long long num_tosses = strtoll(argv[2], NULL, 10);
#endif

	pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
	argument_t* args = (argument_t*)malloc(num_threads * sizeof(argument_t));
	rand_state_t seed = time(NULL);

	for (int i = 0; i < num_threads; ++i) {
		args[i].num_tosses = num_tosses / num_threads
			+ ((i < num_tosses% num_threads) ? 1 : 0);
		args[i].rand_seed = fast_rand(&seed);
		pthread_create(&threads[i], NULL, calculate_pi, (void*)&args[i]);
	}

	long long total_hits = 0;
	for (int i = 0; i < num_threads; ++i) {
		long long num_hits;
		pthread_join(threads[i], (void**)&num_hits);
		total_hits += num_hits;
	}

	printf("%lf\n", 4.0 * total_hits / num_tosses);

	free(args);
	free(threads);
	return 0;
}
