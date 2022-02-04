#pragma once

#include "file_utils.h"
#include "kdtree.h"
#include "tree_printer.h"
#include "utils.h"

#include <mpi.h>
#include <omp.h>
#include <optional>
#include <vector>

inline data_type *read_file_serial(const std::string filename,
                                   std::size_t *size, int *dims) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // we want that only one process is loading the file
  if (rank == 0)
    return read_file(filename, size, dims);
  else
    return nullptr;
}

inline void write_file_serial(const std::string &filename,
                              KNode<data_type> *root, const int dims) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // we want that only one process is loading the file
  if (rank == 0)
    write_file(filename, root, dims);
}

inline double get_time() { return MPI_Wtime(); }

template <typename T> inline void log_message(T obj) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    std::cout << obj << std::endl;
}

#ifdef TEST
inline bool initialize_test(KNode<data_type> *tree, int n_components) {
  std::vector<std::optional<data_type>> *constraints =
      new std::vector<std::optional<data_type>>(n_components * 2);
  for (int i = 0; i < n_components; ++i) {
    // an empty optional
    std::optional<data_type> low;
    (*constraints)[i * 2] = low;

    // an other empty optional
    std::optional<data_type> high;
    (*constraints)[i * 2 + 1] = high;
  }
  return test_kd_tree(tree, constraints, 0);
}

inline bool test(KNode<data_type> *tree, int n_components) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0)
    return initialize_test(tree, n_components);
  else
    return true;
}
#endif
