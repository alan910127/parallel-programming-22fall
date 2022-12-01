#include <mpi.h>

#include "matrix.hh"

/// @brief Release allocated memory
/// @param a_mat pointer to matrix a (n * m integers)
/// @param b_mat pointer to matrix b (m * l integers)
void destruct_matrices(int *a_mat, int *b_mat) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != MASTER) return;

  delete[] a_mat;
  delete[] b_mat;
}
