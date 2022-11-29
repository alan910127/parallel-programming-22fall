#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define MASTER 0

using rand_state_t = long;

int fast_rand(rand_state_t* state) {
  return (((*state = *state * 214013L + 2531011L) >> 16) & 0x7fff);
}

double get_random(rand_state_t* state) {
  return (double)fast_rand(state) / 0x7fff;
}

int main(int argc, char** argv) {
  // --- DON'T TOUCH ---
  MPI_Init(&argc, &argv);
  double start_time = MPI_Wtime();
  double pi_result;
  long long int tosses = atoi(argv[1]);
  int world_rank, world_size;
  // ---

  // MPI init
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  long long int num_tosses =
      tosses / world_size + (world_rank < tosses % world_size ? 1 : 0);

  rand_state_t state = 0xaf * world_rank;
  long long int pi_count = 0;

  for (long long int i = 0; i < num_tosses; ++i) {
    double x = get_random(&state);
    double y = get_random(&state);
    if (x * x + y * y <= 1.0) {
      ++pi_count;
    }
  }

  // binary tree redunction
  int tag = 0;
  MPI_Status status;
  for (int block_size = 1; block_size < world_size; block_size *= 2) {
    int next_block_size = block_size * 2;
    bool should_receive = (world_rank % next_block_size) == 0;
    bool should_send = !should_receive && (world_rank % block_size) == 0;
    if (should_receive) {
      int sender = world_rank + block_size;
      long long int peer_count;
      MPI_Recv(&peer_count, 1, MPI_LONG_LONG_INT, sender, tag, MPI_COMM_WORLD,
               &status);
      pi_count += peer_count;
    } else if (should_send) {
      int receiver = world_rank - block_size;
      MPI_Send(&pi_count, 1, MPI_LONG_LONG_INT, receiver, tag, MPI_COMM_WORLD);
    }
  }

  if (world_rank == 0) {
    // PI result
    pi_result = 4.0 * pi_count / tosses;

    // --- DON'T TOUCH ---
    double end_time = MPI_Wtime();
    printf("%lf\n", pi_result);
    printf("MPI running time: %lf Seconds\n", end_time - start_time);
    // ---
  }

  MPI_Finalize();
  return 0;
}
