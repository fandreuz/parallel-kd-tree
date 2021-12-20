#include <cstring>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <vector>

#if !defined(DOUBLE_PRECISION)
#define mpi_data_type MPI_FLOAT
#define data_type float
#else
#define mpi_data_type MPI_DOUBLE
#define data_type double
#endif

#define TAG_RIGHT_PROCESS_PROCESSING_OVER 10

// represents a data point
class DataPoint {
  data_type *values;
  int data_dimension;

public:
  DataPoint(data_type *dt, int dims) {
    // we need to copy the values since a call to std::nth_element changes the
    // order of the array, therefore pointers do not point anymore to the
    // values we expected
    values = new data_type[dims];
    std::memcpy(values, dt, dims * sizeof(data_type));

    data_dimension = dims;
  }
  ~DataPoint() = delete;

  const data_type get(int index) const {
#ifdef NONSAFE
    if (index < data_dimension)
      return values[index];
    else
      return -1;
#else
    return values[index];
#endif
  }

  data_type *data() { return values; }
};

struct DataPointCompare {
  DataPointCompare(size_t index) : index_(index) {}
  bool operator()(const DataPoint &dp1, const DataPoint &dp2) const {
    return dp1.get(index_) < dp2.get(index_);
  }
  size_t index_;
};

// holds the rank of whoever called this process
int parent = -1;

// number of dimensions in the dataset
int dims;

// rank of this process
int rank;

// maximum process splitting available for the given number of MPI processes
int max_depth;

// list of data point idxes in which this process splitted its branch.
// this process then got assigned the left branch. note that this vector
// contains only "parallel" splits, serial splits are handled otherwise.
std::vector<int> parallel_splits;

DataPoint *serial_splits;

// list of processes started by this process
std::vector<int> children;
// for each child, the size of the branch assigned to that child. used in
// finalize to know what to expect from my children
std::vector<int> right_branch_sizes;
std::vector<int> left_branch_sizes;

int *build_tree(data_type *array, int size, int depth);
int *build_tree_serial(data_type *array, int size, int start_index);
// gather results from all children processes and deliver a complete tree
// to the parent process
int *finalize();

// KNode *as_knode(int *tree) { return nullptr; }

/*
  Generate a kd tree from the given data. If this process is not the main
  process, this function blocks the process until another process wakes
  this process with something to process.

  k = dmd
  data is a 1D array of data such that k consecutive items constitute a data
  point.
  size is the dimension of the dataset, i.e. len(data) / dms.
*/
int *generate_kd_tree(data_type *data, int size, int dms) {
  // we can save dims as a global variable since it is not going to change. it
  // is also constant for all the processes.
  dims = dms;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

  int max_depth = (int)log2(n_processes);

  int depth = 1;
  if (rank != 0) {
    MPI_Status status;

    // receive the number of items in the branch assigned to this process, and
    // the depth of the tree at this point
    int br_size_depth_parent[3];
    MPI_Recv(&br_size_depth_parent, 3, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
             MPI_COMM_WORLD, &status);

    size = br_size_depth_parent[0];
    int depth = br_size_depth_parent[1];
    // rank of the parent which "started" (i.e. waked) this process
    parent = br_size_depth_parent[2];

    data = new data_type[size * dms];
    // receive the data in the branch assigned to this process
    MPI_Recv(data, size * dims, mpi_data_type, MPI_ANY_SOURCE,
             MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  }

  // we create an array which packs all the data in a convenient way
  // this weird mechanic is needed because we do not want to call the default
  // constructor (which the plain 'new' does)
  DataPoint *array = (DataPoint*) ::operator new (sizeof(DataPoint));
  for (int i = 0; i < size; i++) {
    new (array + i) DataPoint(data + i * dims, dims);
  }

  return build_tree(data, size, depth);
}

// sort the given array such that the element in the middle is exactly the
// median with respect to the given axis, and all the items before and
// after are respectively lower/greater than that item.
int sort_and_split(DataPoint *array, int size, int axis) {
  data_type median;
  std::nth_element(array, &median, array + size, DataPointCompare(axis));
  return size / 2;
}

int select_splitting_dimension(int depth) { return (depth - 1) % dims; }

// transform the given DataPoint array in a 1D array such that `dims` contiguous
// items constitute a data point
data_type *unpack_array(DataPoint *array, int size) {
  data_type *unpacked = new data_type[size * dims];
  for (int i = 0; i < size; i++) {
    std::memcpy(unpacked + i * dims, array[i].data, dims * sizeof(int));
  }
  return unpacked;
}

/*
   Construct a tree in a parallel way. The current process takes care of the
   left branch, and delegates the right branch to the process
   rank+2^(D - depth), where D = log2(n_processes).

   This function uses the assumption that we always have 2^k processes, where
   k is a natural number.

   - array is the set of values to be inserted into the tree.
   - size is the size of array
   - depth is the depth of a node created by a call to build_tree. depth starts
    from 1
*/
int *build_tree(DataPoint *array, int size, int depth) {
  int dimension = select_splitting_dimension(depth);
  int split_point_idx = find_split_point(array, size, dimension);

  parallel_splits.push_back(split_point_idx);

  // we hit the bottom line
  if (size <= 1) {
    return finalize();
  } else {
    if (depth > max_depth) {
      serial_splits = new int[size];
      for (int i = 0; i < size; i++)
        serial_splits[i] = 0;

      // depth of the binomial tree resulting from this data
      int right_region = size / 2;

      // right
      build_tree_serial(array + split_point_idx + 1, size - split_point_idx - 1,
                        depth + 1, right_region);
      // left
      return build_tree_serial(array, split_point_idx, depth + 1, 0);
    } else {
      int right_process_rank = rank + pow(2.0, max_depth - depth);
      int right_branch_size = size - split_point_idx - 1;

      data_type *right_branch =
          unpack_array(array + split_point_idx + 1, right_branch_size);

      // we delegate the right part to another process
      MPI_Send(right_branch, right_branch_size, mpi_data_type,
               right_process_rank, 0, MPI_COMM_WORLD, &req);
      delete[] right_branch;

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
int *build_tree_serial(DataPoint *array, int size, int depth, int start_index) {
  int dimension = 0;
  int split_point_idx = find_split_point(array, size, dimension);

  serial_splits[start_index] = split_point_idx;

  if (size <= 1) {
    return finalize();
  } else {
    int right_region = start_index + size / 2;

    // right
    build_tree_serial(array + split_point_idx + 1, size - split_point_idx - 1,
                      depth + 1, right_region);
    // left
    return build_tree_serial(array, split_point_idx, depth + 1,
                             start_index + 1);
  }
}

int *finalize() {
  if (!serial_splits)
    return nullptr;

  // we wait for all the child processes to complete their work
  int n_children = children.size();
  int right_rank = -1, right_branch_size = -1, left_branch_size = -1,
      split_idx = -1;
  // buffer which contains the split indexes from the right branch
  int *right_branch_buffer = nullptr;
  int *left_branch_buffer = serial_splits;
  serial_splits = nullptr;

  // merged_array contains the values which results from merging a right branch
  // with a left branch.
  int *merging_array;
  for (int i = n_children - 1; i > 0; --i) {
    right_rank = children.at(i);
    right_branch_size = right_branch_sizes.at(i);
    left_branch_size = left_branch_sizes.at(i);
    split_idx = parallel_splits.at(i);

    merging_array =
        new data_type[(right_branch_size + left_branch_size + 1) * dims];
    right_branch_buffer = new data_type[right_branch_size * dims];

    MPI_Status status;
    MPI_Recv(right_branch_buffer, right_branch_size, mpi_data_type, right_rank,
             TAG_RIGHT_PROCESS_PROCESSING_OVER, MPI_COMM_WORLD, &status);

    // the root of this tree is the data point used to split left and right
    merging_array[0] = split_idx;
    for (int dpth = 0; dpth < (int)log2(left_branch_size); ++dpth) {
      // number of nodes at the current level in the left/right subtree
      int n_of_nodes = pow(2.0, (double)dpth - 1);

      // we put into the three what's inside the left subtree
      std::memcpy(merging_array + 1, left_branch_buffer,
                  n_of_nodes * dims * sizeof(data_type));
      // we put into the three what's inside the right subtree
      std::memcpy(merging_array + n_of_nodes + 1, right_branch_buffer,
                  n_of_nodes * dims * sizeof(data_type));
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
    MPI_Send(left_branch_buffer,
             (right_branch_size + left_branch_size + 1) * dims, mpi_data_type,
             parent, TAG_RIGHT_PROCESS_PROCESSING_OVER, MPI_COMM_WORLD);
    delete[] left_branch_buffer;
    return nullptr;
  } else {
    // this is the root process
    return left_branch_buffer;
  }
}
