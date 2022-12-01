#include <mpi.h>
#include <stdio.h>

#include "matrix.hh"

/// @brief Read size of matrix a and matrix b (n, m, l) and data of matrices
/// from stdin
/// @param n_ptr pointer to n
/// @param m_ptr pointer to m
/// @param l_ptr pointer to l
/// @param a_mat_ptr pointer to matrix a (continuous memory space for placing
/// n * m elements of int)
/// @param b_mat_ptr pointer to matrix b (continuous memory space for placing
/// m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr,
                        int **b_mat_ptr) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != MASTER) return;

  int_ref n = *n_ptr, m = *m_ptr, l = *l_ptr;
  mat_ref a_mat = *a_mat_ptr, b_mat = *b_mat_ptr;

  scanf("%d %d %d", n_ptr, m_ptr, l_ptr);

  a_mat = new int[n * m];
  b_mat = new int[m * l];

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      scanf("%d", &a_mat[i * m + j]);
    }
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < l; ++j) {
      scanf("%d", &b_mat[i * l + j]);
    }
  }
}