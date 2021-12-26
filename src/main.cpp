#include "file_reader.h"
#include "tree_mpi.h"

#include <mpi.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string filename = argc > 1 ? argv[1] : "../benchmark/benchmark1.csv";

  int SIZE = -1, DIMS = -1;
  data_type *dt = nullptr;

  if (rank == 0)
    dt = read_file(filename, &SIZE, &DIMS);

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
