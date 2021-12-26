#ifndef TREE_H
#define TREE_H

#include <iostream>
#include <limits>

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

template <typename T> class KNode {
  // point used to split the tree
  T *data;
  int dims;

  // left and right branch originating from this node
  KNode<T> *left = nullptr, *right = nullptr;

  // if this node is the root, when deleting the KNode we should also free
  // the variable data, which is an array that in fact holds all the data
  // stored in this KNode.
  // after that we should not delete anything else in children
  bool is_root = false;

  void print_node_values(std::ostream &os) const {
    os << "(";
    for (int i = 0; i < dims; i++) {
      if (i > 0)
        os << ",";
      if (data[i] == std::numeric_limits<int>::min()) {
        os << "n/a";
        break;
      } else
        os << data[i];
    }
    os << ")";
  }

  void print_tree(std::ostream &os, const std::string &prefix,
                  bool isLeft) const {
    os << prefix;

    os << (isLeft ? "├──" : "└──");

    // print the value of the node
    print_node_values(os);
    os << std::endl;

    // enter the next tree level - left and right branch
    if (left)
      left->print_tree(os, prefix + (isLeft ? "│   " : "    "), true);
    if (right)
      right->print_tree(os, prefix + (isLeft ? "│   " : "    "), false);
  }

public:
  KNode(data_type *d, int dms, KNode<T> *l, KNode<T> *r, bool root) {
    data = d;
    dims = dms;
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

  friend std::ostream &operator<<(std::ostream &os, const KNode<T> &node) {
    node.print_tree(os, "", false);
    return os;
  }
};

#endif // TREE_H
