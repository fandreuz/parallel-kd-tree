#include "tree_mpi.h"
#include <iostream>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  data_type dt[100];
  for (int i = 0; i < 100; i++) {
    dt[i] = i * i - 2 * i;
  }

  data_type *tree = generate_kd_tree(dt, 100, 2);

  MPI_Finalize();
}
