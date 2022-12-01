#include <mpi.h>

#include "matrix.hh"

static inline void print_matrix(const mat_t &mat, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%d ", mat[i * cols + j]);
    }
    putchar('\n');
  }
}

/// @brief Matrix multiplication (also output the result)
/// @param n Number of rows of matrix a
/// @param m Number of columns of matrix a / number of rows of matrix b
/// @param l Number of columns of matrix b
/// @param a_mat a continuous memory placing n * m elements of int
/// @param b_mata continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l, const int *a_mat,
                     const int *b_mat) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}