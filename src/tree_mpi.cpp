#include "tree.h"
#include <iostream>
#include <math.h>
#include <vector>

#define TAG_RIGHT_PROCESS_PROCESSING_OVER 10

// holds the rank of whoever called this process
int parent = -1;

// list of data point idxes in which this process splitted its branch.
// this process then got assigned the left branch. note that this vector
// contains only "parallel" splits, serial splits are handled otherwise.
std::vector<int> parallel_splits;

int *serial_splits;

// list of processes started by this process
std::vector<int> children;
// for each child, the size of the branch assigned to that child. used in
// finalize to know what to expect from my children
std::vector<int> right_branch_sizes;
std::vector<int> left_branch_sizes;

void build_tree(data_type *array, int size);
// gather results from all children processes and deliver a complete tree
// to the parent process
void finalize();

int main() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

  if (rank != 0) {
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
  } else {
    // root process
    data_type data[100];
    for (int i = 0; i < 100; i++) {
      data[i] = i * i - 2 * i;
    }
    int *tree = build_tree(data, 100);

    for (int i = 0; i < 100; i++) {
      std::cout << tree[i] << std::endl;
    }

    delete[] tree;
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
int *build_tree(data_type *array, int size, int depth) {
  int dimension = 0;
  int split_point_idx = find_split_point(array, size, dimension);

  splits.push_back(split_point_idx);

  // we hit the bottom line
  if (size <= 1) {
    return finalize();
  } else {
    int n_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    int max_depth = (int)log2(n_processes);
    if (depth > max_depth) {
      serial_splits = new int[size];

      // depth of the binomial tree resulting from this data
      int right_region = size / 2;

      // right
      build_tree_serial(array + split_point_idx + 1, size - split_point_idx - 1,
                        serial_splits, right_region);
      // left
      return build_tree_serial(array, split_point_idx, 0);
    } else {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      int right_process_rank = rank + pow(2.0, max_depth - depth);
      int right_branch_size = size - split_point_idx - 1;

      // we delegate the right part to another process
      MPI_Request req;
      MPI_Isend(array + split_point_idx + 1, right_branch_size, mpi_data_type,
                right_process_rank, 0, MPI_COMM_WORLD, &req);

      children.push_back(right_process_rank);
      right_branch_sizes.push_back(right_branch_size);
      left_branch_sizes.push_back(split_point_idx);

      // this process takes care of the left part
      return build_tree(array, split_point_idx, depth + 1);
    }
  }
}

/*
   Construct a tree in a serial way. The current process takes care of both
   the left and right branch.

   - array is the set of values to be inserted into the tree.
   - size is the size of array
*/
int *build_tree_serial(data_type *array, int size, int start_index) {
  int dimension = 0;
  int split_point_idx = find_split_point(array, size, dimension);

  splits.push_back(split_point_idx);

  if (size <= 1) {
    return finalize();
  } else {
    int right_region = start_index + size / 2;

    // right
    build_tree_serial(array + split_point_idx + 1, size - split_point_idx - 1,
                      right_region);
    // left
    return build_tree_serial(array, split_point_idx, start_index);
  }
}

int *finalize() {
  if (!serial_splits)
    return;

  // we wait for all the child processes to complete their work
  int n_children = children.size();
  int right_rank = -1, right_branch_size = -1, left_branch_size = -1,
      split_idx = -1;
  // buffer which contains the split indexes from the right branch
  int *right_branch_buffer = nullptr;
  int *left_branch_buffer = serial_splits;
  serial_splits = nullptr;

  // contain the merged split indexes which results from merging a right
  // branch with a left branch
  int *merging_array;
  for (int i = n_children - 1; i > 0; --i) {
    right_rank = children.at(i);
    right_branch_size = right_branch_sizes.at(i);
    left_branch_size = left_branch_sizes.at(i);
    split_idx = splits.at(i);

    merging_array = new int[right_branch_size + left_branch_size + 1];
    right_branch_buffer = new int[right_branch_size];

    MPI_Status status;
    MPI_Recv(buffer, right_branch_size, MPI_INT, right_rank,
             TAG_RIGHT_PROCESS_PROCESSING_OVER, MPI_COMM_WORLD, &status);

    // the root of this tree is the data point used to split left and right
    merging_array[0] = split_idx;
    for (int dpth = 0; dpth < log2(left_branch_size); ++dpth) {
      // number of nodes at the current level in the left/right subtree
      int n_of_nodes = pow(2.0, (double)dpth - 1);

      // we put into the three what's inside the left subtree
      std::memcpy(merging_array, left_branch_buffer, n_of_nodes);
      // we put into the three what's inside the right subtree
      std::memcpy(merging_array + n_of_nodes, right_branch_buffer, n_of_nodes);
    }

    delete[] right_branch_buffer;
    delete[] left_branch_buffer;
    // we go one level up, therefore the merging array is now the array that
    // represents the left branch buffer
    left_branch_buffer = merging_array;
  }

  if (parent != -1) {
    // we finished merging left and right parallel subtrees, we can contact the
    // parent and transfer the data
    MPI_Send(left_branch_buffer, splits.size(), MPI_INT, parent,
             TAG_RIGHT_PROCESS_PROCESSING_OVER, MPI_COMM_WORLD);
    delete[] left_branch_buffer;
  } else {
    // this is the root process
    return left_branch_buffer;
  }
}
