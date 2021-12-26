#include "tree_mpi.h"
#include <mpi.h>

#include <iostream>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int SIZE = -1, DIMS = -1;

  if (getenv("KDTREE_SIZE") == NULL)
    SIZE = 10;
  else
    SIZE = atoi(getenv("KDTREE_SIZE"));
  if (getenv("KDTREE_DIMS") == NULL)
    DIMS = 3;
  else
    DIMS = atoi(getenv("KDTREE_DIMS"));

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  data_type *dt = nullptr;
  if (rank == 0) {
    dt = new data_type[SIZE * DIMS];

    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < DIMS; j++) {
        dt[i * DIMS + j] = i - 2 * j;
      }
    }

#ifdef DEBUG
    for (int i = 0; i < SIZE * DIMS; i++) {
      if (i % DIMS == 0)
        std::cout << "(";
      std::cout << dt[i];
      if (i % DIMS == DIMS - 1) {
        std::cout << ")";
        if (i < SIZE * DIMS - 1)
          std::cout << " / ";
      } else
        std::cout << ",";
    }
    std::cout << std::endl;
#endif
  }

#ifdef TIME
  double start_time = MPI_Wtime();
#endif

  KNode<data_type> *tree = generate_kd_tree(dt, SIZE, DIMS);

#ifdef TIME
  if (rank == 0) {
    double end_time = MPI_Wtime();
    std::cout << "# " << end_time - start_time << std::endl;
  }
#endif

  // we can now delete the data safely
  delete[] dt;

#ifdef OUTPUT
  if (rank == 0) {
    std::cout << *tree;
  }
#endif

  delete tree;

  MPI_Finalize();
}
