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

class KNode {
  // point used to split the tree
  data_type *data;
  // left and right branch originating from this node
  KNode *left = nullptr, *right = nullptr;

  // if this node is the root, when deleting the KNode we should also free
  // the variable data, which is an array that in fact holds all the data
  // stored in this KNode.
  // after that we should not delete anything else in children
  bool is_root = false;

public:
  KNode(data_type *d, KNode *l, KNode *r, bool root) {
    data = d;
    left = l;
    right = r;
    is_root = root;
  }

  ~KNode() {
    if (is_root)
      delete[] data;
    delete left;
    delete right;
  }

  data_type get_data(int i) { return data[i]; }
  KNode *get_left() { return left; }
  KNode *get_right() { return right; }
};

#endif // TREE_H
