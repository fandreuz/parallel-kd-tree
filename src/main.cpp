#include "tree_mpi.h"
#include <iostream>

#define SIZE 3
#define DIMS 2

void print(data_type *tree);

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  data_type *dt = nullptr;
  if (rank == 0) {
    dt = new data_type[SIZE*DIMS];

    for (int i = 0; i < SIZE*DIMS; i++) {
      dt[i] = i * i - 2 * i;

#ifdef DEBUG
      std::cout << i << " -> " << dt[i] << std::endl;
#endif
    }
  }

  data_type *tree = generate_kd_tree(dt, SIZE, DIMS);
  int next_powersum = 1;
  int current_multiplier = 1;
  if (rank == 0)
  {
    for (int i = 0; i < SIZE; i++) {
      std::cout << "(";
      for (int j = 0; j < DIMS; j++) {
        if (j > 0)
          std::cout << ",";
        std::cout << tree[i * DIMS + j];
      }
      std::cout << ")";

      if(i + 1 == next_powersum) {
        std::cout << "--";
        current_multiplier *= 2;
        next_powersum += current_multiplier;
      }
    }
  }

  delete[] dt;

  MPI_Finalize();
}
