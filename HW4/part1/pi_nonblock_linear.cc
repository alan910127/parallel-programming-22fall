#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
  // --- DON'T TOUCH ---
  MPI_Init(&argc, &argv);
  double start_time = MPI_Wtime();
  double pi_result;
  long long int tosses = atoi(argv[1]);
  int world_rank, world_size;
  // ---

  // TODO: MPI init

  if (world_rank > 0) {
    // TODO: MPI workers
  } else if (world_rank == 0) {
    // TODO: non-blocking MPI communication.
    // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
    MPI_Request requests[];

    MPI_Waitall();
  }

  if (world_rank == 0) {
    // TODO: PI result

    // --- DON'T TOUCH ---
    double end_time = MPI_Wtime();
    printf("%lf\n", pi_result);
    printf("MPI running time: %lf Seconds\n", end_time - start_time);
    // ---
  }

  MPI_Finalize();
  return 0;
}
