#include "file_reader.h"
#include "tree_printer.h"

#ifdef USE_MPI
#include "kdtree_mpi.h"
#include <mpi.h>
#else
#include "kdtree_openmp.h"
#include <omp.h>
#endif

int main(int argc, char **argv) {
  const std::string filename =
      argc > 1 ? argv[1] : "../benchmark/benchmark1.csv";

  // size and number of components per data point in the dataset
  int SIZE = -1, DIMS = -1;
  // the dataset as a 1D array, DIMS consecutive items of dt are a data point
  data_type *dt = nullptr;

#ifdef USE_MPI
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // we want that only one process is loading the file
  if (rank == 0)
    dt = read_file(filename, &SIZE, &DIMS);
#else
  // if we're using OpenMP there's no need to check that this is the main
  // process
  dt = read_file(filename, &SIZE, &DIMS);
#endif

#ifdef TIME
#ifdef USE_MPI
  double start_time = MPI_Wtime();
#else
  double start_time = omp_get_wtime();
#endif
#endif

  KNode<data_type> *tree = generate_kd_tree(dt, SIZE, DIMS);

#ifdef TIME
#ifdef USE_MPI
  if (rank == 0) {
    double end_time = MPI_Wtime();
    std::cout << "# " << end_time - start_time << std::endl;
  }
#else
  double end_time = omp_get_wtime();
  std::cout << "# " << end_time - start_time << std::endl;
#endif
#endif

  // we can now delete the data safely
  delete[] dt;

#ifdef OUTPUT
#ifdef USE_MPI
  if (rank == 0) {
    std::cout << *tree;
  }
#else
  std::cout << *tree;
#endif
#endif

  delete tree;

#ifdef USE_MPI
  MPI_Finalize();
#endif
}
