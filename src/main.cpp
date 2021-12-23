#include "tree_mpi.h"
#include <iostream>

#define SIZE 6
#define DIMS 2

void print(data_type *tree);

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  data_type *dt = nullptr;
  if (rank == 0) {
    dt = new data_type[SIZE * DIMS];

    for (int i = 0; i < SIZE; i++) {
      dt[i * 2] = 9 - i;
      dt[i * 2 + 1] = 1 + i;
    }

    for (int i = 0; i < SIZE * DIMS; i++) {
#ifdef DEBUG
      std::cout << i << " -> " << dt[i] << std::endl;
#endif
    }
  }

  int new_size;
  int *new_size_ptr = nullptr;
  if (rank == 0)
    new_size_ptr = &new_size;

  data_type *tree = generate_kd_tree(dt, SIZE, DIMS, new_size_ptr);

  int next_powersum = 1;
  int current_multiplier = 1;
  int counter = 0;
  if (rank == 0) {
    for (int i = 0; i < *new_size_ptr; i++) {
      std::cout << "(";
      for (int j = 0; j < DIMS; j++) {
        if (j > 0)
          std::cout << ",";
        std::cout << tree[i * DIMS + j];
      }
      std::cout << ")";

      if (i + 1 == next_powersum) {
        std::cout << "--";
        current_multiplier *= 2;
        next_powersum += current_multiplier;
        counter = 0;
      } else {
        counter++;
        if (counter == 2) {
          std::cout << " | ";
          counter = 0;
        }
      }
    }
  }

  delete[] dt;

  MPI_Finalize();
}
