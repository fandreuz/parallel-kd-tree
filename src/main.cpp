#include "tree_mpi.h"
#include <iostream>

#define SIZE 6
#define DIMS 2

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  data_type dt[SIZE];
  for (int i = 0; i < SIZE; i++) {
    dt[i] = i * i - 2 * i;
  }

  data_type *tree = generate_kd_tree(dt, SIZE, DIMS);

  MPI_Finalize();
}
