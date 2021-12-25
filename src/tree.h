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

template <typename T>
class KNode {
  // point used to split the tree
  T *data;
  // left and right branch originating from this node
  KNode<T> *left = nullptr, *right = nullptr;

  // if this node is the root, when deleting the KNode we should also free
  // the variable data, which is an array that in fact holds all the data
  // stored in this KNode.
  // after that we should not delete anything else in children
  bool is_root = false;

public:
  KNode(data_type *d, KNode<T> *l, KNode<T> *r, bool root) {
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

  const T get_data(int i) const { return data[i]; }
  const KNode<T> *get_left() const { return left; }
  const KNode<T> *get_right() const { return right; }
};

#endif // TREE_H
