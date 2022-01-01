#include "tree_mpi.h"

#if !defined(DOUBLE_PRECISION)
#define mpi_data_type MPI_FLOAT
#define data_type float
#else
#define mpi_data_type MPI_DOUBLE
#define data_type double
#endif

#define TAG_RIGHT_PROCESS_PROCESSING_OVER 10
#define TAG_RIGHT_PROCESS_N_ITEMS 11

// rank of the parent process of this process
int parent = -1;

// number of components for each data point
int dims = -1;

// rank of this process
int rank = -1;

// number of MPI processes available
int n_processes = -1;

// maximum depth of the tree at which we can parallelize. after this depth no
// more right-branches can be assigned to non-surplus processes
int max_depth = 0;

// number of additional processes that are not enough to parallelize an entire
// level of the tree, they are assigned left-to-right until there are no more
// surplus processes
int surplus_processes = 0;

// number of items assigned serially (i.e. non-parallelizable) to this process
int serial_branch_size = 0;

// DataPoint used to split a branch assigned to this process. this process then
// received the left branch resulting from the split.
std::vector<DataPoint> parallel_splits;

// DataPoints in the serial branch assigned to this process. see also
// build_tree_serial
DataPoint *serial_splits;

// if the i-th item is true, the i-th item in serial_splits is initialized
bool *initialized;

// children of this process, i.e. processes that received a right branch from
// this process
std::vector<int> children;

void build_tree(DataPoint *array, int size, int depth);
void build_tree_serial(DataPoint *array, int size, int depth, int region_width,
                       int region_start_index, int branch_starting_index);
data_type *finalize(int &new_size);

KNode<data_type> *generate_kd_tree(data_type *data, int size, int dms) {
  // we can save dims as a global variable since it is not going to change. it
  // is also constant for all the processes.
  dims = dms;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef MPI_DEBUG
  int debug_rank = atoi(getenv("MPI_DEBUG_RANK"));
  std::cerr << "MPI_DEBUG_RANK=" << atoi(getenv("MPI_DEBUG_RANK")) << std::endl;
  if (rank == debug_rank) {
    volatile int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }
#endif

  MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

  max_depth = log2((double)n_processes);
  surplus_processes = n_processes - (int)pow(2.0, (double)max_depth);
#ifdef DEBUG
  if (rank == 0) {
    std::cout << "Starting " << n_processes << " with max_depth = " << max_depth
              << std::endl;
  }

  std::cout << "[rank" << rank << "]: started" << std::endl;
#endif

  int depth = 0;
  if (rank != 0) {
    MPI_Status status;

    // receive the number of items in the branch assigned to this process, and
    // the depth of the tree at this point
    int br_size_depth_parent[4];
    MPI_Recv(&br_size_depth_parent, 4, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
             MPI_COMM_WORLD, &status);

    // number of data points in the branch
    size = br_size_depth_parent[0];
    if (size == 0) {
      // a process warned this process that there is no work to perform
      return nullptr;
    }

    // depth of the tree at this point
    depth = br_size_depth_parent[1];
    // rank of the parent which "started" (i.e. waked) this process
    parent = br_size_depth_parent[2];
    // dimensionality of the data points
    dims = br_size_depth_parent[3];

#ifdef DEBUG
    std::cout << "[rank" << rank << "]: went to sleep" << std::endl;
#endif

    data = new data_type[size * dims];
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

  // we can delete data if and only if we're the owner, i.e. we created the
  // data, but this is not true if the rank is 0 (in such case the data is
  // owned by the user).
  if (rank != 0)
    delete[] data;

#ifdef DEBUG
  std::cout << "[rank" << rank
            << "]: starting parallel build_tree (branch size: " << size << ")"
            << std::endl;
#endif

  build_tree(array, size, depth);
  // size might be changed by finalize (the actual size of the tree may not
  // be equal to the original size of the dataset)
  data_type *tree = finalize(size);
  return convert_to_knodes(tree, size, dims, 0, 1, 0);
}

/*
   Sort the given array such that the element in the middle is exactly the
   median with respect to the given axis, and all the items before and
   after are respectively lower/greater than that item.
*/
inline int sort_and_split(DataPoint *array, int size, int axis) {
  // the second part of median_idx is needed to unbalance the split towards the
  // left region (which is the one which may parallelize with the highest
  // probability).
  int median_idx = size / 2 - 1 * ((size + 1) % 2);
  std::nth_element(array, array + median_idx, array + size,
                   DataPointCompare(axis));
  // if size is 2 we want to return the first element (the smallest one), since
  // it will be placed into the first empty spot in serial_split
  return median_idx;
}

inline int select_splitting_dimension(int depth) { return depth % dims; }

inline int next_process_rank(int next_depth) {
  return rank + pow(2.0, max_depth - next_depth);
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
    from 0
*/
void build_tree(DataPoint *array, int size, int depth) {
  int next_depth = depth + 1;

  if (size <= 1 || next_depth > max_depth + 1 ||
      (next_depth == max_depth + 1 && rank >= surplus_processes)) {
#ifdef DEBUG
    if (size <= 1)
      std::cout << "[rank" << rank << "]: hit the bottom! " << std::endl;
    else {
      std::cout << "[rank" << rank
                << "]: no available processes, going serial from now "
                << std::endl;
    }
#endif
    if (size > 0) {
      // we want that the serial branch is storable in an array whose size is
      // a powersum of two
      serial_branch_size = bigger_powersum_of_two(size);
      serial_splits =
          (DataPoint *)::operator new(serial_branch_size * sizeof(DataPoint));

      initialized = new bool[serial_branch_size];
      for (int i = 0; i < serial_branch_size; ++i) {
        initialized[i] = false;
      }

      build_tree_serial(array, size, depth, 1, 0, 0);
    }

    // this process should have called a surplus process to do some stuff, but
    // since we have only one or less items in the buffer we could not call
    // anyone. however we need to wake that process to avoid deadlock
    if (size <= 1 && next_depth == max_depth + 1 && rank < surplus_processes) {
      int right_process_rank = n_processes - surplus_processes + rank;

      int right_branch_data[4];
      right_branch_data[0] = 0;
      MPI_Send(right_branch_data, 4, MPI_INT, right_process_rank, 0,
               MPI_COMM_WORLD);
    }
  } else {
    int dimension = select_splitting_dimension(depth);
    int split_point_idx = sort_and_split(array, size, dimension);

#ifdef DEBUG
    std::cout << "[rank" << rank << "]: parallel split against axis "
              << dimension << ", split_idx = " << split_point_idx << std::endl;
#endif

    parallel_splits.push_back(std::move(array[split_point_idx]));

    int right_process_rank =
        (next_depth < max_depth + 1) * next_process_rank(next_depth) +
        (next_depth == max_depth + 1) * (rank < surplus_processes) *
            (n_processes - surplus_processes + rank);
    int right_branch_size = size - split_point_idx - 1;

#ifdef DEBUG
    std::cout << "[rank" << rank
              << "]: delegating right region (starting from) "
              << split_point_idx + 1 << " (size " << right_branch_size << " of "
              << size << ") to rank" << right_process_rank << std::endl;
#endif

    int right_branch_data[4];
    right_branch_data[0] = right_branch_size;
    right_branch_data[1] = next_depth;
    right_branch_data[2] = rank;
    right_branch_data[3] = dims;
    MPI_Send(right_branch_data, 4, MPI_INT, right_process_rank, 0,
             MPI_COMM_WORLD);

    data_type *right_branch =
        unpack_array(array + split_point_idx + 1, right_branch_size, dims);

    // we delegate the right part to another process
    // this is synchronous since we also want to delete the buffer ASAP
    MPI_Send(right_branch, right_branch_size * dims, mpi_data_type,
             right_process_rank, 0, MPI_COMM_WORLD);
    delete[] right_branch;

    children.push_back(right_process_rank);

    // if there is nothing left in this branch we need to artificially augument
    // it with a fictious node
    data_type *fake_data;
    if (split_point_idx == 0) {
      fake_data = new data_type[dims];
      for (int i = 0; i < dims; ++i)
        fake_data[i] = EMPTY_PLACEHOLDER;
      array = new DataPoint(fake_data, dims);
      delete[] fake_data;
      // since this (local) variale is used as the size in the next call to
      // build_tree we increase it by one (since we generated fake data).
      split_point_idx = 1;
    }
    // this process takes care of the left part
    build_tree(array, split_point_idx, next_depth);

    /*
    if (split_point_idx == 0) {
      delete[] fake_data;
      delete array;
    }
    */
  }
}

/*
   Construct a tree in a serial way. The current process takes care of both
   the left and right branch.

   - array is the set of values to be inserted into the tree.
   - size is the size of array
   - depth is the depth of the tree after the addition of this new level
   - region_width is the width of the current region of serial_splits which
      holds the current level of the serial tree
   - region_start_index is the index in of serial_splits in which the region
      corresponding to the current level starts;
   - branch_starting_index is the index of serial_splits (starting from
      region_width) in which the item used to split this branch is stored.
*/
void build_tree_serial(DataPoint *array, int size, int depth, int region_width,
                       int region_start_index, int branch_starting_index) {
  initialized[region_start_index + branch_starting_index] = true;

  if (size <= 1) {
#ifdef DEBUG
    std::cout << "[rank" << rank << "]: hit the bottom! " << std::endl;
#endif
    new (serial_splits + region_start_index + branch_starting_index)
        DataPoint(std::move(array[0]));
  } else {
    int dimension = select_splitting_dimension(depth);
    int split_point_idx = sort_and_split(array, size, dimension);

#ifdef DEBUG
    std::cout << "[rank" << rank << "]: serial split against axis " << dimension
              << ", split_idx = " << split_point_idx << ", size = " << size
              << std::endl;
#endif

    new (serial_splits + region_start_index + branch_starting_index)
        DataPoint(std::move(array[split_point_idx]));

    // we update the values for the next iteration
    region_start_index += region_width;
    region_width *= 2;
    branch_starting_index *= 2;

    // right
    build_tree_serial(array + split_point_idx + 1, size - split_point_idx - 1,
                      depth + 1, region_width, region_start_index,
                      branch_starting_index + 1);
    // left
    if (split_point_idx > 0)
      build_tree_serial(array, split_point_idx, depth + 1, region_width,
                        region_start_index, branch_starting_index);
  }
}

data_type *finalize(int &size) {
  // we wait for all the child processes to complete their work
  int n_children = children.size();

#ifdef DEBUG
  std::cout << "[rank" << rank
            << "]: finalize called. #children = " << n_children << std::endl;
#endif

  int right_rank = -1, right_branch_size = -1;
  // buffer which contains the split indexes from the right branch
  data_type *right_branch_buffer = nullptr;

  data_type *left_branch_buffer = nullptr;
  int left_branch_size = serial_branch_size;

  if (serial_branch_size > 0) {
    left_branch_buffer = unpack_risky_array(serial_splits, serial_branch_size,
                                            dims, initialized);
  }

  // merged_array contains the values which results from merging a right branch
  // with a left branch.
  data_type *merging_array;
  for (int i = n_children - 1; i >= 0; --i) {
    right_rank = children.at(i);

    MPI_Status status;
    MPI_Recv(&right_branch_size, 1, MPI_INT, right_rank,
             TAG_RIGHT_PROCESS_N_ITEMS, MPI_COMM_WORLD, &status);

    right_branch_buffer = new data_type[right_branch_size * dims];

    // we gather the branch from another process
    MPI_Recv(right_branch_buffer, right_branch_size * dims, mpi_data_type,
             right_rank, TAG_RIGHT_PROCESS_PROCESSING_OVER, MPI_COMM_WORLD,
             &status);

    int branch_size = left_branch_size;
    if (right_branch_size != left_branch_size) {
      int max = std::max(right_branch_size, left_branch_size);
      int min = std::min(right_branch_size, left_branch_size);

      data_type *old_buffer =
          min == left_branch_size ? left_branch_buffer : right_branch_buffer;

      data_type *temp = new data_type[max * dims];
      std::memcpy(temp, old_buffer, min * dims * sizeof(data_type));
      for (int i = min * dims; i < max * dims; ++i) {
        temp[i] = EMPTY_PLACEHOLDER;
      }

      delete[] old_buffer;

      branch_size = max;

      if (left_branch_size < right_branch_size) {
        left_branch_buffer = temp;
      } else {
        right_branch_buffer = temp;
      }
    }

    DataPoint split_item = std::move(parallel_splits.at(i));

    merging_array = new data_type[(branch_size * 2 + 1) * dims];

    // the root of this tree is the data point used to split left and right
    std::memcpy(merging_array, split_item.data(), dims * sizeof(data_type));

    rearrange_branches(merging_array + dims, left_branch_buffer,
                       right_branch_buffer, branch_size, dims);

    delete[] right_branch_buffer;
    delete[] left_branch_buffer;

    // we go one level up, therefore the merging array is now the array that
    // represents the left branch buffer
    left_branch_buffer = merging_array;

    // the new size of the left branch is the sum of the former left branch size
    // and of the right branch size, plus 1 (the split point)
    left_branch_size = branch_size * 2 + 1;
  }

  if (parent != -1) {
    // we finished merging left and right parallel subtrees, we can contact
    // the parent and transfer the data

    // first of all the number of data points transmitted
    MPI_Send(&left_branch_size, 1, MPI_INT, parent, TAG_RIGHT_PROCESS_N_ITEMS,
             MPI_COMM_WORLD);

    MPI_Send(left_branch_buffer, left_branch_size * dims, mpi_data_type, parent,
             TAG_RIGHT_PROCESS_PROCESSING_OVER, MPI_COMM_WORLD);
    delete[] left_branch_buffer;
    return nullptr;
  } else {
    // this is the root process
    size = left_branch_size;
    return left_branch_buffer;
  }
}
