#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <vector>

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

  int tag = 0;
  if (world_rank > MASTER) {
    // MPI workers
    MPI_Request request;
    MPI_Isend(&pi_count, 1, MPI_LONG_LONG_INT, MASTER, tag, MPI_COMM_WORLD,
              &request);
    MPI_Request_free(&request);  // No need to perform wait
  } else if (world_rank == MASTER) {
    // non-blocking MPI communication.
    // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
    int num_requests = world_size - 1;
    std::vector<MPI_Request> requests(num_requests);
    std::vector<MPI_Status> status(num_requests);
    std::vector<long long int> peer_count(num_requests);

    for (int peer = 1; peer < world_size; ++peer) {
      MPI_Irecv(&peer_count[peer - 1], 1, MPI_LONG_LONG_INT, peer, tag,
                MPI_COMM_WORLD, &requests[peer - 1]);
    }

    MPI_Waitall(num_requests, requests.data(), status.data());

    for (auto& count : peer_count) {
      pi_count += count;
    }
  }

  if (world_rank == MASTER) {
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
