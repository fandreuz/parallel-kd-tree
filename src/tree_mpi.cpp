#include "tree.h"
#include <iostream>
#include <math.h>
#include <vector>

#define TAG_PROCESS_PROCESSING_OVER 10

// holds the rank of whoever called this process
int parent = -1;

// list of data point idxes in which this process split its branch
std::vector<int> splits;

// list of processes started by this process
std::vector<int> children;

void build_tree(data_type *array, int size);
// gather results from all children processes and deliver a complete tree
// to the parent process
void finalize();

int main() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

  if (rank == 0) {
    data_type data[100];
    for (int i = 0; i < 100; i++) {
      data[i] = i * i - 2 * i;
    }
    build_tree(data, 100);
  } else {
    // receive the number of items in the branch assigned to this process, and
    // the depth of the tree at this point
    int br_size_depth_parent[3];
    MPI_Recv(&br_size_depth_parent, 3, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
             MPI_COMM_WORLD, requests + n_processes * (i - 1));
    int branch_size = br_size_depth_parent[0];
    int depth = br_size_depth_parent[1];
    parent = br_size_depth_parent[2];

    data_type *data = new data_type[branch_size];
    // receive the data in the branch assigned to this process
    MPI_Recv(data, branch_size, mpi_data_type, MPI_ANY_SOURCE, MPI_ANY_TAG,
             MPI_COMM_WORLD, requests + n_processes * (i - 1) - 1);

    build_tree(data, branch_size, depth, splits);
  }
}

int find_split_point(data_type *array, int size, int dimension) { return 0; }

/*
   Construct a tree in a parallel way. The current process takes care of the
   left branch, and delegates the right branch to the process
   rank+2^(D - depth), where D = log2(n_processes).

   This function uses the assumption that we always have 2^k processes, where
   k is a natural number.

   - array is the set of values to be inserted into the tree.
   - size is the size of array
   - depth is the depth of a node created by a call to build_tree. depth starts
    from one
*/
void build_tree(data_type *array, int size, int depth) {
  int dimension = 0;
  int split_point_idx = find_split_point(array, size, dimension);

  splits.push_back(split_point_idx);

  // we hit the bottom line
  if (size <= 1) {
      finalize();
  }
  else
  {
      int n_processes;
      MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

      int max_depth = (int)log2(n_processes);
      if (depth > max_depth)
      {
          // left
          build_tree_serial(array, split_point_idx, splits);
          // right
          build_tree_serial(array + split_point_idx + 1, size - split_point_idx - 1,
                            splits);
      } else {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      int right_process_rank = rank + pow(2.0, max_depth - depth);

      // we delegate the right part to another process
      MPI_Request req;
      MPI_Isend(array + split_point_idx + 1, size - split_point_idx - 1,
                mpi_data_type, right_process_rank, 0, MPI_COMM_WORLD, &req);

      // this process takes care of the left part
      build_tree(array, split_point_idx, depth + 1);
    }
  }
}

/*
   Construct a tree in a serial way. The current process takes care of both
   the left and right branch.

   - array is the set of values to be inserted into the tree.
   - size is the size of array
*/
void build_tree_serial(data_type *array, int size) {
  int dimension = 0;
  int split_point_idx = find_split_point(array, size, dimension);

  splits.push_back(split_point_idx);

  if (size <= 1) {
      finalize();
  }
  else
  {
      // left
      build_tree_serial(array, split_point_idx, splits);
      // right
      build_tree_serial(array + split_point_idx + 1, size - split_point_idx - 1,
                        splits);
  }
}

void finalize() {

}
