#ifndef TREE_H
#define TREE_H

#ifdef MPI_VERSION
#include <mpi.h>
#endif

#if !defined(DOUBLE_PRECISION)
#define data_type float

#ifdef MPI_VERSION
#define mpi_data_type MPI_FLOAT
#endif

#else
#define data_type double

#ifdef MPI_VERSION
#define mpi_data_type MPI_FLOAT
#endif

#endif

struct KNode {
  // point used to split the tree
  data_type *data;
  // left and right branch originating from this node
  KNode *left = nullptr, *right = nullptr;
};

#endif // TREE_H
