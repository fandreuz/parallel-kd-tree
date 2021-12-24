#include "tree_mpi.h"
#include <iostream>
#include <limits>

#define SIZE 6
#define DIMS 2

void print(data_type *tree, int size);

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

  int size = SIZE;
  data_type *tree = generate_kd_tree(dt, size, DIMS);

  if (rank == 0) {
    delete[] dt;
    print(tree, size);
    delete[] tree;
  }

  MPI_Finalize();
}

void print(data_type *tree, int size) {
  int next_powersum = 1;
  int current_multiplier = 1;
  int counter = 0;
  for (int i = 0; i < size; i++) {
    std::cout << "(";
    for (int j = 0; j < DIMS; j++) {
      if (j > 0)
        std::cout << ",";

      if (tree[i * DIMS + j] == std::numeric_limits<int>::min()) {
        std::cout << "n/a";
        break;
      } else
        std::cout << tree[i * DIMS + j];
    }
    std::cout << ")";

    if (i + 1 == next_powersum) {
      std::cout << std::endl;
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
