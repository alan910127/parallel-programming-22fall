#include <mpi.h>

#include <vector>

#include "matrix.hh"

static inline void print_matrix(const int *mat, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%d ", mat[i * cols + j]);
    }
    putchar('\n');
  }
}

enum Tags { DIMENSION = 1, A_MAT = 2, B_MAT = 3, C_MAT = 4 };

namespace Matmul {
int size, rank;

void master(int n, int m, int l, const int *a_mat, const int *b_mat) {
  int *c_mat = new int[n * l];

  std::vector<MPI_Request> dim_requests(size);
  std::vector<MPI_Request> mat_requests(2 * size);
  std::vector<MPI_Status> dim_statuses(size);
  std::vector<MPI_Status> mat_statuses(2 * size);
  std::vector<int[3]> dimensions(size);

  int average_rows = n / size;
  int extra_rows = n % size;
  int start_row = dimensions[0][0] =
      (rank < extra_rows) ? average_rows + 1 : average_rows;
  dimensions[0][1] = m;
  dimensions[0][2] = l;

  for (int i = 1; i < size; ++i) {
    int num_rows = (i < extra_rows) ? average_rows + 1 : average_rows;
    dimensions[i][0] = num_rows;
    dimensions[i][1] = m;
    dimensions[i][2] = l;
    MPI_Isend(&dimensions[i], 3, MPI_INT, i, Tags::DIMENSION, MPI_COMM_WORLD,
              &dim_requests[i]);
    MPI_Isend(&a_mat[start_row * m], num_rows * m, MPI_INT, i, Tags::A_MAT,
              MPI_COMM_WORLD, &mat_requests[2 * i]);
    MPI_Isend(b_mat, m * l, MPI_INT, i, Tags::B_MAT, MPI_COMM_WORLD,
              &mat_requests[2 * i + 1]);
    start_row += num_rows;
  }

  MPI_Waitall(size - 1, dim_requests.data() + 1, dim_statuses.data() + 1);

  int master_n = dimensions[0][0];
  for (int k = 0; k < l; ++k) {
    for (int i = 0; i < master_n; ++i) {
      c_mat[i * l + k] = 0;
      for (int j = 0; j < m; ++j) {
        c_mat[i * l + k] += a_mat[i * m + j] * b_mat[j * l + k];
      }
    }
  }

  MPI_Waitall(2 * (size - 1), mat_requests.data() + 2, mat_statuses.data() + 2);

  std::vector<MPI_Request> c_mat_requests(size);
  std::vector<MPI_Status> c_mat_statuses(size);

  start_row = dimensions[0][0];
  for (int i = 1; i < size; ++i) {
    int num_rows = dimensions[i][0];
    MPI_Irecv(&c_mat[start_row * l], num_rows * l, MPI_INT, i, Tags::C_MAT,
              MPI_COMM_WORLD, &c_mat_requests[i]);
    start_row += num_rows;
  }

  MPI_Waitall(size - 1, c_mat_requests.data() + 1, c_mat_statuses.data() + 1);

  print_matrix(c_mat, n, l);
  delete[] c_mat;
}

void worker() {
  int dimensions[3];
  MPI_Status status;
  MPI_Recv(dimensions, 3, MPI_INT, MASTER, Tags::DIMENSION, MPI_COMM_WORLD,
           &status);
  auto [n, m, l] = dimensions;
  int *a_mat = new int[n * m];
  int *b_mat = new int[m * l];
  int *c_mat = new int[n * l];

  MPI_Request requests[2];
  MPI_Status statuses[2];
  MPI_Irecv(a_mat, n * m, MPI_INT, MASTER, Tags::A_MAT, MPI_COMM_WORLD,
            &requests[0]);
  MPI_Irecv(b_mat, m * l, MPI_INT, MASTER, Tags::B_MAT, MPI_COMM_WORLD,
            &requests[1]);
  MPI_Waitall(2, requests, statuses);

  for (int k = 0; k < l; ++k) {
    for (int i = 0; i < n; ++i) {
      c_mat[i * l + k] = 0;
      for (int j = 0; j < m; ++j) {
        c_mat[i * l + k] += a_mat[i * m + j] * b_mat[j * l + k];
      }
    }
  }

  MPI_Send(c_mat, n * l, MPI_INT, MASTER, Tags::C_MAT, MPI_COMM_WORLD);

  delete[] a_mat;
  delete[] b_mat;
  delete[] c_mat;
}

void run(int n, int m, int l, const int *a_mat, const int *b_mat) {
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == MASTER) {
    master(n, m, l, a_mat, b_mat);
  } else {
    worker();
  }
}
}  // namespace Matmul

/// @brief Matrix multiplication (also output the result)
/// @param n Number of rows of matrix a
/// @param m Number of columns of matrix a / number of rows of matrix b
/// @param l Number of columns of matrix b
/// @param a_mat a continuous memory placing n * m elements of int
/// @param b_mata continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l, const int *a_mat,
                     const int *b_mat) {
  Matmul::run(n, m, l, a_mat, b_mat);
}