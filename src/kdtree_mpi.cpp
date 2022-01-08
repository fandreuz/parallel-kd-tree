#include "kdtree_mpi.h"

/**
 * @def
 * @brief Communicate to the parent MPI process that we're over with the branch
 *          assigned to this process, and that we are sending back the results.
 */
#define TAG_RIGHT_PROCESS_PROCESSING_OVER 10
/**
 * @def
 * @brief Communicate to the parent MPI process the number of data point we are
 *          going to send.
 */
#define TAG_RIGHT_PROCESS_N_ITEMS 11
/**
 * @def
 * @brief Communicate to a child process the data points in the branch it is
 *          assigned to. Attached to this communication there should be some
 *          info regarding the branch (number of data points, depth of the tree
 *          at this point, rank of the parent, number of components in the
 *          data points).
 */
#define TAG_RIGHT_PROCESS_START 12

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
std::optional<DataPoint> *serial_splits = nullptr;

// children of this process, i.e. processes that received a right branch from
// this process
std::vector<int> children;

void build_tree(std::vector<DataPoint>::iterator first_data_point,
                std::vector<DataPoint>::iterator end_data_point, int size,
                int depth);
void build_tree_serial(std::vector<DataPoint>::iterator first_data_point,
                       std::vector<DataPoint>::iterator end_data_point,
                       int depth, int region_width, int region_start_index,
                       int branch_starting_index);
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

  max_depth = compute_max_depth(n_processes);
  surplus_processes = compute_n_surplus_processes(n_processes, max_depth);
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
    MPI_Recv(&br_size_depth_parent, 4, MPI_INT, MPI_ANY_SOURCE,
             TAG_RIGHT_PROCESS_START, MPI_COMM_WORLD, &status);

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

  std::vector<DataPoint> data_points;
  data_points.reserve(size);
  for (int i = 0; i < size; i++) {
    data_points.push_back(DataPoint(data + i * dims, dims));
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

  build_tree(data_points.begin(), data_points.end(), size, depth);

  // size might be changed by finalize (the actual size of the tree may not
  // be equal to the original size of the dataset)
  data_type *tree = finalize(size);
  if (rank != 0)
    return nullptr;

  return convert_to_knodes(tree, size, dims, 0, 1, 0);
}

/*
   Construct a tree in a parallel way. The current process takes care of the
   left branch, and delegates the right branch to the process
   rank+2^(D - depth), where D = log2(n_processes).

   This function uses the assumption that we always have 2^k processes, where
   k is a natural number.

   - first_data_point is an iterator pointing to the first data point in the
      set;
   - end_data_point is an iterator pointing to past-the-last data point;
   - depth is the depth of a node created by a call to build_tree. depth starts
    from 0
*/
void build_tree(std::vector<DataPoint>::iterator first_data_point,
                std::vector<DataPoint>::iterator end_data_point, int size,
                int depth) {
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
      // sum of them are NOT going to be initialized since they are placeholders
      // of leafs (last level of the tree) that are not present since
      // size < bigger_powersum_of_two
      serial_splits = new std::optional<DataPoint>[serial_branch_size];

      build_tree_serial(first_data_point, end_data_point, depth, 1, 0, 0);
    }

    // this process should have called a surplus process to do some stuff, but
    // since we have only one or less items in the buffer we could not call
    // anyone. however we need to wake that process to avoid deadlock
    if (size <= 1 && next_depth == max_depth + 1 && rank < surplus_processes) {
      int right_process_rank = n_processes - surplus_processes + rank;

      int right_branch_data[4];
      right_branch_data[0] = 0;
      MPI_Send(right_branch_data, 4, MPI_INT, right_process_rank,
               TAG_RIGHT_PROCESS_START, MPI_COMM_WORLD);
    }
  } else {
    int dimension = select_splitting_dimension(depth, dims);
    int split_point_idx =
        sort_and_split(first_data_point, end_data_point, dimension);

#ifdef DEBUG
    std::cout << "[rank" << rank << "]: parallel split against axis "
              << dimension << ", split_idx = " << split_point_idx << std::endl;
#endif

    parallel_splits.push_back(std::move(*(first_data_point + split_point_idx)));

    int right_process_rank = compute_next_process_rank(
        rank, max_depth, next_depth, surplus_processes, n_processes);
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
    MPI_Send(right_branch_data, 4, MPI_INT, right_process_rank,
             TAG_RIGHT_PROCESS_START, MPI_COMM_WORLD);

    std::vector<DataPoint>::iterator right_branch_first_point =
        first_data_point + split_point_idx + 1;
    data_type *right_branch =
        unpack_array(right_branch_first_point, end_data_point, dims);

    // we delegate the right part to another process
    // this is synchronous since we also want to delete the buffer ASAP
    MPI_Send(right_branch, right_branch_size * dims, mpi_data_type,
             right_process_rank, 0, MPI_COMM_WORLD);
    delete[] right_branch;

    children.push_back(right_process_rank);

    if (split_point_idx != 0) {
      // this process takes care of the left part
      build_tree(first_data_point, right_branch_first_point - 1,
                 split_point_idx, next_depth);
    }
  }
}

/*
   Construct a tree serially. The current process takes care of both the left
   and right branch.

   A region (i.e. k contiguous elements) of serial_splits holds an entire level
   of the k-d tree (i.e. elements whose distance from the root is the same).

   - first_data_point is an iterator pointing to the first data point in the
      set;
   - end_data_point is an iterator pointing to past-the-last data point;
   - depth is the depth of the tree after the addition of this new level;
   - region_width is the width of the current region of serial_splits which
      holds the current level of the serial tree. this increases (multiplied
      by 2) at each recursive call;
   - region_start_index is the index of serial_splits in which the region
      corresponding to the current level starts (i.e. the index after the end
      of the region corresponding to the level before);
   - branch_starting_index is the index of serial_splits (starting from
      region_width) in which the item used to split this branch is stored.
*/
void build_tree_serial(std::vector<DataPoint>::iterator first_data_point,
                       std::vector<DataPoint>::iterator end_data_point,
                       int depth, int region_width, int region_start_index,
                       int branch_starting_index) {
  // this is equivalent to say that there is at most one data point in the
  // sequence
  if (first_data_point == end_data_point ||
      first_data_point + 1 == end_data_point) {
#ifdef DEBUG
    std::cout << "[rank" << rank << "]: hit the bottom! " << std::endl;
#endif
    serial_splits[region_start_index + branch_starting_index].emplace(
        DataPoint(std::move(*first_data_point)));
  } else {
    int dimension = select_splitting_dimension(depth, dims);
    int split_point_idx =
        sort_and_split(first_data_point, end_data_point, dimension);

#ifdef DEBUG
    std::cout << "[rank" << rank << "]: serial split against axis " << dimension
              << ", split_idx = " << split_point_idx
              << ", size = " << std::distance(first_data_point, end_data_point);
    << std::endl;
#endif

    serial_splits[region_start_index + branch_starting_index].emplace(
        DataPoint(std::move(*(first_data_point + split_point_idx))));

    // we update the values for the next iteration
    region_start_index += region_width;
    region_width *= 2;
    branch_starting_index *= 2;
    depth += 1;

    std::vector<DataPoint>::iterator right_branch_first_point =
        first_data_point + split_point_idx + 1;
    // right
    build_tree_serial(right_branch_first_point, end_data_point, depth,
                      region_width, region_start_index,
                      branch_starting_index + 1);
    // left
    if (split_point_idx > 0)
      build_tree_serial(first_data_point, right_branch_first_point - 1, depth,
                        region_width, region_start_index,
                        branch_starting_index);
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
    left_branch_buffer = unpack_optional_array(
        serial_splits, serial_branch_size, dims, EMPTY_PLACEHOLDER);
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
