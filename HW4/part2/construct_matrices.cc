#include <mpi.h>
#include <stdio.h>

#include "matrix.hh"

constexpr int MAXLINE = 1 << 20;

static inline char readchar() {
  static char buf[MAXLINE], *p = buf, *q = buf;
  if (p == q && (q = buf) + fread(buf, 1, MAXLINE, stdin)) == buf) return EOF;
  return *p++;
}

static inline int next_int() {
  int x = 0;
  char c = readchar();
  while (('0' > c || c > '9') && c != EOF) c = readchar();
  while ('0' <= c && c <= '9') {
    x = x * 10 + (c ^ '0');
    c = readchar();
  }
  return x;
}

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

  n = next_int();
  m = next_int();
  l = next_int();

  a_mat = new int[n * m];
  b_mat = new int[m * l];

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      a_mat[i * m + j] = next_int();
    }
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < l; ++j) {
      b_mat[i * l + j] = next_int();
    }
  }
}