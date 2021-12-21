#include "tree_mpi.h"

#include <unistd.h>

#if !defined(DOUBLE_PRECISION)
#define mpi_data_type MPI_FLOAT
#define data_type float
#else
#define mpi_data_type MPI_DOUBLE
#define data_type double
#endif

#define TAG_RIGHT_PROCESS_PROCESSING_OVER 10

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

// this is an array of pointers since DataPoint resulting from serial splits
// are taken from an already existing DataPoint array
DataPoint *serial_splits;

// list of processes started by this process
std::vector<int> children;
// for each child, the size of the branch assigned to that child. used in
// finalize to know what to expect from my children
std::vector<int> right_branch_sizes;
std::vector<int> left_branch_sizes;

/*
  Generate a kd tree from the given data. If this process is not the main
  process, this function blocks the process until another process wakes
  this process with something to process.

  k = dmd
  data is a 1D array of data such that k consecutive items constitute a data
  point.
  size is the dimension of the dataset, i.e. len(data) / dms.
*/
data_type *generate_kd_tree(data_type *data, int size, int dms) {
  // we can save dims as a global variable since it is not going to change. it
  // is also constant for all the processes.
  dims = dms;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef MPI_DEBUG
  if (rank == atoi(getenv("MPI_DEBUG_RANK"))) {
    volatile int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }
#endif

  int n_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

  int max_depth = (int)log2(n_processes);
#ifdef DEBUG
  if (rank == 0) {
    std::cout << "Starting " << n_processes << " with max_depth = " << max_depth
              << std::endl;
  }

  std::cout << "[rank" << rank << "]: started" << std::endl;
#endif

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

#ifdef DEBUG
    std::cout << "[rank" << rank << "]: went to sleep" << std::endl;
#endif

    data = new data_type[size * dms];
    // receive the data in the branch assigned to this process
    MPI_Recv(data, size * dims, mpi_data_type, MPI_ANY_SOURCE, MPI_ANY_TAG,
             MPI_COMM_WORLD, &status);

#ifdef DEBUG
    std::cout << "[rank" << rank << "]: waked by rank" << parent << std::endl;
#endif
  }

  // we create an array which packs all the data in a convenient way.
  // this weird mechanic is needed because we do not want to call the default
  // constructor (which the plain 'new' does)
  DataPoint *array = (DataPoint *)::operator new(size * sizeof(DataPoint));
  for (int i = 0; i < size; i++) {
    new (array + i) DataPoint(data + i * dims, dims);
  }

#ifdef DEBUG
  std::cout << "[rank" << rank
            << "]: starting parallel build_tree (branch size: " << size << ")"
            << std::endl;
#endif

  return build_tree(array, size, depth);
}

// sort the given array such that the element in the middle is exactly the
// median with respect to the given axis, and all the items before and
// after are respectively lower/greater than that item.
int sort_and_split(DataPoint *array, int size, int axis) {
  std::nth_element(array, array + size / 2, array + size,
                   DataPointCompare(axis));
  return size / 2;
}

int select_splitting_dimension(int depth) { return (depth - 1) % dims; }

// transform the given DataPoint array in a 1D array such that `dims` contiguous
// items constitute a data point
data_type *unpack_array(DataPoint *array, int size) {
  data_type *unpacked = new data_type[size * dims];
  for (int i = 0; i < size; i++) {
    std::memcpy(unpacked + i * dims, array[i].data(), dims * sizeof(data_type));
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
data_type *build_tree(DataPoint *array, int size, int depth) {
  // we hit the bottom line
  if (size <= 1) {
#ifdef DEBUG
    std::cout << "[rank" << rank << "]: hit the bottom! " << std::endl;
#endif
    // we can call finalize() from there since per each process there is goint
    // to be only one "active" call to build_tree (the parent recursive calls
    // are inactive in the sense that as soon as the children build_tree()
    // returns they are going to return too).
    return finalize();
  } else {
    if (depth > max_depth) {
#ifdef DEBUG
      std::cout << "[rank" << rank
                << "]: no available processes, going serial from now "
                << std::endl;
#endif
      serial_splits = (DataPoint *)::operator new(size * sizeof(DataPoint));
      left_branch_sizes.push_back(size);
      build_tree_serial(array, size, depth, 0);
      return finalize();
    } else {
      int dimension = select_splitting_dimension(depth);
      int split_point_idx = sort_and_split(array, size, dimension);

#ifdef DEBUG
      std::cout << "[rank" << rank << "]: parallel split against axis "
                << dimension << ", split_idx = " << split_point_idx
                << std::endl;
#endif

      parallel_splits.push_back(split_point_idx);

      int right_process_rank = rank + pow(2.0, max_depth - depth);
      int right_branch_size = size - split_point_idx - 1;

#ifdef DEBUG
      std::cout << "[rank" << rank
                << "]: delegating right region (starting from) "
                << split_point_idx + 1 << " (size " << right_branch_size
                << " of " << size << ") to rank" << right_process_rank
                << std::endl;
#endif

      data_type *right_branch =
          unpack_array(array + split_point_idx + 1, right_branch_size);

      // we delegate the right part to another process
      // this is synchronous since we also want to delete the buffer ASAP
      MPI_Send(right_branch, right_branch_size, mpi_data_type,
               right_process_rank, 0, MPI_COMM_WORLD);
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
void build_tree_serial(DataPoint *array, int size, int depth, int start_index) {
  if (size <= 1) {
#ifdef DEBUG
    std::cout << "[rank" << rank << "]: hit the bottom! " << std::endl;
#endif
    return;
  } else {
    int dimension = select_splitting_dimension(depth);
    int split_point_idx = sort_and_split(array, size, dimension);

#ifdef DEBUG
    std::cout << "[rank" << rank << "]: serial split against axis " << dimension
              << ", split_idx = " << split_point_idx << ", size = " << size
              << std::endl;
#endif

    new (serial_splits + start_index)
        DataPoint(std::move(array[split_point_idx]));

    int right_region = start_index + size / 2;

    // right
    build_tree_serial(array + split_point_idx + 1, size - split_point_idx - 1,
                      depth + 1, right_region);
    // left
    build_tree_serial(array, split_point_idx, depth + 1, start_index + 1);
  }
}

data_type *finalize() {
  if (!serial_splits)
    return nullptr;

  // we wait for all the child processes to complete their work
  int n_children = children.size();

#ifdef DEBUG
  std::cout << "[rank" << rank
            << "]: finalize called. #children = " << n_children << std::endl;
#endif

  int right_rank = -1, right_branch_size = -1,
      left_branch_size = left_branch_sizes.at(n_children), split_idx = -1;
  // buffer which contains the split indexes from the right branch
  data_type *right_branch_buffer = nullptr;
  data_type *left_branch_buffer = unpack_array(serial_splits, left_branch_size);
  serial_splits = nullptr;

  // merged_array contains the values which results from merging a right branch
  // with a left branch.
  data_type *merging_array;
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
