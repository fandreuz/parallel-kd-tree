#pragma once

#include "file_utils.h"
#include "tree_printer.h"

#include "kdtree.h"

#ifdef USE_MPI
#include <mpi.h>
#else
#include <omp.h>
#endif

inline data_type *read_file_serial(const std::string filename,
                                   std::size_t *size, int *dims) {
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // we want that only one process is loading the file
  if (rank == 0)
    return read_file(filename, size, dims);
  else
    return nullptr;
#else
  // if we're using OpenMP there's no need to check that this is the main
  // process
  return read_file(filename, size, dims);
#endif
}

inline void write_file_serial(const std::string &filename,
                              KNode<data_type> *root, const int dims) {
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // we want that only one process is loading the file
  if (rank == 0)
    write_file(filename, root, dims);
#else
  // if we're using OpenMP there's no need to check that this is the main
  // process
  write_file(filename, root, dims);
#endif
}

inline double get_time() {
#ifdef USE_MPI
  return MPI_Wtime();
#else
  return omp_get_wtime();
#endif
}

template <typename T> inline void log_message(T obj) {
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    std::cout << obj;
#else
  std::cout << obj;
#endif
}

inline void init_parallel_environment(int *argc, char ***argv) {
#ifdef USE_MPI
  MPI_Init(argc, argv);
#endif
}

inline void finalize_parallel_environment() {
#ifdef USE_MPI
#ifdef DEBUG
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout << "[rank" << rank << "] finalizing" << std::endl;
#endif

  MPI_Finalize();
#endif
}
