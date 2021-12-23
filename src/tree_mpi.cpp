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

// holds the rank of whoever called this process
int parent = -1;

// number of dimensions in the dataset
int dims;

// rank of this process
int rank;

// number of MPI processes
int n_processes;

// maximum process splitting available for the given number of MPI processes
int max_depth = 0;

// n of processes such that rank > 2^max_depth
int surplus_processes;

int serial_branch_size = 0;

// list of data point idxes in which this process splitted its branch.
// this process then got assigned the left branch. note that this vector
// contains only "parallel" splits, serial splits are handled otherwise.
std::vector<DataPoint> parallel_splits;

// this is an array of pointers since DataPoint resulting from serial splits
// are taken from an already existing DataPoint array
DataPoint *serial_splits;
// if an index is true, the corresponding index in serial_splits contains an
// item which was initialized
bool *initialized;

// list of processes started by this process
std::vector<int> children;

/*
  Generate a kd tree from the given data. If this process is not the main
  process, this function blocks the process until another process wakes
  this process with something to process.

  k = dmd
  data is a 1D array of data such that k consecutive items constitute a data
  point.
  size is the dimension of the dataset, i.e. len(data) / dms.
*/
data_type *generate_kd_tree(data_type *data, int &size, int dms) {
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
    int br_size_depth_parent[3];
    MPI_Recv(&br_size_depth_parent, 3, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
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
  return finalize(size);
}

/*
   Sort the given array such that the element in the middle is exactly the
   median with respect to the given axis, and all the items before and
   after are respectively lower/greater than that item.
*/
inline int sort_and_split(DataPoint *array, int size, int axis) {
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

      build_tree_serial(array, size, depth, 0, serial_branch_size);
    }

    // this process should have called a surplus process to do some stuff, but
    // since we have only one or less items in the buffer we could not call
    // anyone. however we need to wake that process to avoid deadlock
    if (size <= 1 && next_depth == max_depth + 1 && rank < surplus_processes) {
      int right_process_rank = n_processes - surplus_processes + rank;

      int right_branch_data[3];
      right_branch_data[0] = 0;
      MPI_Send(right_branch_data, 3, MPI_INT, right_process_rank, 0,
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

    int right_branch_data[3];
    right_branch_data[0] = right_branch_size;
    right_branch_data[1] = next_depth;
    right_branch_data[2] = rank;
    MPI_Send(right_branch_data, 3, MPI_INT, right_process_rank, 0,
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

    if (split_point_idx == 0) {
      delete[] fake_data;
      delete array;
    }
  }
}

/*
   Construct a tree in a serial way. The current process takes care of both
   the left and right branch.

   - array is the set of values to be inserted into the tree.
   - size is the size of array
   - depth is the depth of the tree after the addition of this new level
   - start_index is the first index of serial_splits in which we can write
      something
   - right_limit is the first element on the right in which we cannot write
*/
void build_tree_serial(DataPoint *array, int size, int depth, int start_index,
                       int right_limit) {
  initialized[start_index] = true;

  if (size <= 1) {
#ifdef DEBUG
    std::cout << "[rank" << rank << "]: hit the bottom! " << std::endl;
#endif
    new (serial_splits + start_index) DataPoint(std::move(array[0]));
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

    // we can start writing the left region immediately after the cell in which
    // we wrote the split point
    int regions_size = (right_limit - start_index - 1) / 2;
    int left_region = start_index + 1;
    int right_region = left_region + regions_size;

    // right
    build_tree_serial(array + split_point_idx + 1, size - split_point_idx - 1,
                      depth + 1, right_region, right_limit);
    // left
    if (split_point_idx > 0)
      build_tree_serial(array, split_point_idx, depth + 1, left_region,
                        right_region);
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
    left_branch_buffer = new data_type[serial_branch_size];
    // this is a temp copy used to keep the data safe
    data_type *temp_left_branch_buffer = unpack_risky_array(
        serial_splits + 1, serial_branch_size - 1, dims, initialized + 1);

    // we copy the first serial splitting item into left_branch_buffer
    std::memcpy(left_branch_buffer, serial_splits[0].data(),
                dims * sizeof(data_type));

    int branches_size = (serial_branch_size - 1) / 2;
    // we skip the first element since it is going to stay there
    rearrange_branches(
        left_branch_buffer + dims, temp_left_branch_buffer, branches_size,
        temp_left_branch_buffer + branches_size * dims, branches_size, dims);

    // TODO: this fails for some reason I do not understand...
    // delete[] temp_left_branch_buffer;
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

    if (right_branch_size != left_branch_size) {
      int max = std::max(right_branch_size, left_branch_size);
      int min = std::min(right_branch_size, left_branch_size);

      data_type *old_buffer =
          min == left_branch_size ? left_branch_buffer : right_branch_buffer;

      data_type *temp = new data_type[max * dims];
      std::memcpy(temp, old_buffer, min * dims * sizeof(data_type));
      for (int i = min * dims; i < max * dims; i++) {
        temp[i] = EMPTY_PLACEHOLDER;
      }

      if (left_branch_size < right_branch_size) {
        delete[] left_branch_buffer;
        left_branch_buffer = temp;
        left_branch_size = max;
      }
      else
      {
        delete[] right_branch_buffer;
        right_branch_buffer = temp;
        right_branch_size = max;
      }
    }

    DataPoint split_item = std::move(parallel_splits.at(i));

    merging_array =
        new data_type[(right_branch_size + left_branch_size + 1) * dims];

    // the root of this tree is the data point used to split left and right
    std::memcpy(merging_array, split_item.data(), dims * sizeof(data_type));

    rearrange_branches(merging_array + dims, left_branch_buffer,
                       left_branch_size, right_branch_buffer, right_branch_size,
                       dims);

    // TODO: this fails for some reason I do not understand...
    // delete[] right_branch_buffer;
    // delete[] left_branch_buffer;
    // we go one level up, therefore the merging array is now the array that
    // represents the left branch buffer
    left_branch_buffer = merging_array;

    // the new size of the left branch is the sum of the former left branch size
    // and of the right branch size, plus 1 (the split point)
    left_branch_size += right_branch_size + 1;
  }

  if (parent != -1) {
    // we finished merging left and right parallel subtrees, we can contact
    // the parent and transfer the data

    // first of all the number of data points transmitted
    MPI_Send(&left_branch_size, 1, MPI_INT, parent, TAG_RIGHT_PROCESS_N_ITEMS,
             MPI_COMM_WORLD);

    MPI_Send(left_branch_buffer, left_branch_size * dims, mpi_data_type, parent,
             TAG_RIGHT_PROCESS_PROCESSING_OVER, MPI_COMM_WORLD);
    // TODO: this fails for some reason I do not understand...
    // delete[] left_branch_buffer;
    return nullptr;
  } else {
    // this is the root process
    size = left_branch_size;
    return left_branch_buffer;
  }
}
