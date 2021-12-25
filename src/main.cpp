#include "tree_mpi.h"
#include <mpi.h>

#include <iostream>
#include <limits>

void print(const KNode<data_type> *node);
void print(const std::string &prefix, const KNode<data_type> *node,
           bool isLeft);

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

  double start_time = MPI_Wtime();
  KNode<data_type> *tree = generate_kd_tree(dt, SIZE, DIMS);
  double end_time = MPI_Wtime();

  // we can now delete the data safely
  delete[] dt;

#ifdef OUTPUT
  if (rank == 0) {
    print(tree);
  }
#endif

  delete tree;

#ifdef TIME
  if (rank == 0) {
    std::cout << "# " << end_time - start_time << std::endl;
  }
#endif

  MPI_Finalize();
}

void print_node(const KNode<data_type> *node) {
  std::cout << "(";
  for (int i = 0; i < dims; i++) {
    if (i > 0)
      std::cout << ",";
    if (node->get_data(i) == std::numeric_limits<int>::min()) {
      std::cout << "n/a";
      break;
    } else
      std::cout << node->get_data(i);
  }
  std::cout << ")";
}

// implementation taken from https://stackoverflow.com/a/51730733/6585348
void print(const std::string &prefix, const KNode<data_type> *node,
           bool isLeft) {
  if (node != nullptr) {
    std::cout << prefix;

    std::cout << (isLeft ? "├──" : "└──");

    // print the value of the node
    print_node(node);
    std::cout << std::endl;

    // enter the next tree level - left and right branch
    print(prefix + (isLeft ? "│   " : "    "), node->get_left(), true);
    print(prefix + (isLeft ? "│   " : "    "), node->get_right(), false);
  }
}

void print(const KNode<data_type> *node) { print("", node, false); }
