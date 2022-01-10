#include "kdtree.h"

data_type *KDTreeGreenhouse::retrieve_dataset_info() {
#ifdef DEBUG
  std::cout << "[rank" << rank << "]: went to sleep" << std::endl;
#endif

  MPI_Status status;

  // receive the number of items in the branch assigned to this process, and
  // the depth of the tree at this point
  int br_size_depth_parent[4];
  MPI_Recv(&br_size_depth_parent, 4, MPI_INT, MPI_ANY_SOURCE,
           TAG_RIGHT_PROCESS_START, MPI_COMM_WORLD, &status);

  // number of data points in the branch
  n_datapoints = br_size_depth_parent[0];
  // depth of the tree at this point
  starting_depth = br_size_depth_parent[1];
  // rank of the parent which "started" (i.e. waked) this process
  parent = br_size_depth_parent[2];
  n_components = br_size_depth_parent[3];

  data_type *data = new data_type[n_datapoints * n_components];
  // receive the data in the branch assigned to this process
  MPI_Recv(data, n_datapoints * n_components, mpi_data_type, MPI_ANY_SOURCE,
           MPI_ANY_TAG, MPI_COMM_WORLD, &status);

#ifdef DEBUG
  std::cout << "[rank" << rank << "]: waked by rank" << parent << std::endl;
#endif

  return data;
}

/*
   Construct a tree in a parallel way. The current process takes care of the
   left branch, and delegates the right branch to the process
   rank+2^(D - depth), where D = log2(n_parallel_workers).

   This function uses the assumption that we always have 2^k processes, where
   k is a natural number.

   - first_data_point is an iterator pointing to the first data point in the
      set;
   - end_data_point is an iterator pointing to past-the-last data point;
   - depth is the depth of a node created by a call to build_tree. depth starts
    from 0
*/
void KDTreeGreenhouse::build_tree(
    std::vector<DataPoint>::iterator first_data_point,
    std::vector<DataPoint>::iterator end_data_point, int depth) {
  int next_depth = depth + 1;

  if (n_datapoints <= 1 || next_depth > max_depth + 1 ||
      (next_depth == max_depth + 1 && rank >= surplus_processes)) {
#ifdef DEBUG
    if (n_datapoints <= 1)
      std::cout << "[rank" << rank << "]: hit the bottom! " << std::endl;
    else {
      std::cout << "[rank" << rank
                << "]: no available processes, going serial from now "
                << std::endl;
    }
#endif
    if (n_datapoints > 0) {
      // we want that the serial branch is storable in an array whose size is
      // a powersum of two
      serial_tree_size = bigger_powersum_of_two(n_datapoints);
      // sum of them are NOT going to be initialized since they are placeholders
      // of leafs (last level of the tree) that are not present since
      // n_datapoints < bigger_powersum_of_two
      serial_tree = new std::optional<DataPoint>[serial_tree_size];

      build_tree_serial(first_data_point, end_data_point, depth, 1, 0, 0);
    }

    // this process should have called a surplus process to do some stuff, but
    // since we have only one or less items in the buffer we could not call
    // anyone. however we need to wake that process to avoid deadlock
    if (n_datapoints <= 1 && next_depth == max_depth + 1 &&
        rank < surplus_processes) {
      int right_process_rank = n_parallel_workers - surplus_processes + rank;

      int right_branch_data[4];
      right_branch_data[0] = 0;
      MPI_Send(right_branch_data, 4, MPI_INT, right_process_rank,
               TAG_RIGHT_PROCESS_START, MPI_COMM_WORLD);
    }
  } else {
    int dimension = select_splitting_dimension(depth, n_components);
    int split_point_idx =
        sort_and_split(first_data_point, end_data_point, dimension);

#ifdef DEBUG
    std::cout << "[rank" << rank << "]: parallel split against axis "
              << dimension << ", split_idx = " << split_point_idx << std::endl;
#endif

    parallel_splits.push_back(std::move(*(first_data_point + split_point_idx)));

    int right_process_rank = compute_next_process_rank(
        rank, max_depth, next_depth, surplus_processes, n_parallel_workers);
    int right_branch_size = n_datapoints - split_point_idx - 1;

#ifdef DEBUG
    std::cout << "[rank" << rank
              << "]: delegating right region (starting from) "
              << split_point_idx + 1 << " (size " << right_branch_size << " of "
              << n_datapoints << ") to rank" << right_process_rank << std::endl;
#endif

    int right_branch_data[4];
    right_branch_data[0] = right_branch_size;
    right_branch_data[1] = next_depth;
    right_branch_data[2] = rank;
    right_branch_data[3] = n_components;
    MPI_Send(right_branch_data, 4, MPI_INT, right_process_rank,
             TAG_RIGHT_PROCESS_START, MPI_COMM_WORLD);

    std::vector<DataPoint>::iterator right_branch_first_point =
        first_data_point + split_point_idx + 1;
    data_type *right_branch =
        unpack_array(right_branch_first_point, end_data_point, n_components);

    // we delegate the right part to another process
    // this is synchronous since we also want to delete the buffer ASAP
    MPI_Send(right_branch, right_branch_size * n_components, mpi_data_type,
             right_process_rank, 0, MPI_COMM_WORLD);
    delete[] right_branch;

    n_datapoints = split_point_idx;

    children.push_back(right_process_rank);

    if (split_point_idx != 0) {
      // this process takes care of the left part
      build_tree(first_data_point, right_branch_first_point - 1, next_depth);
    }
  }
}

data_type *KDTreeGreenhouse::finalize(int *kdtree_size) {
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
  int left_branch_size = serial_tree_size;

  if (serial_tree_size > 0) {
    left_branch_buffer = unpack_optional_array(serial_tree, serial_tree_size,
                                               n_components, EMPTY_PLACEHOLDER);
  }

  // merged_array contains the values which results from merging a right branch
  // with a left branch.
  data_type *merging_array;
  for (int i = n_children - 1; i >= 0; --i) {
    right_rank = children.at(i);

    MPI_Status status;
    MPI_Recv(&right_branch_size, 1, MPI_INT, right_rank,
             TAG_RIGHT_PROCESS_N_ITEMS, MPI_COMM_WORLD, &status);

    right_branch_buffer = new data_type[right_branch_size * n_components];

    // we gather the branch from another process
    MPI_Recv(right_branch_buffer, right_branch_size * n_components,
             mpi_data_type, right_rank, TAG_RIGHT_PROCESS_PROCESSING_OVER,
             MPI_COMM_WORLD, &status);

    int branch_size = left_branch_size;
    if (right_branch_size != left_branch_size) {
      int max = std::max(right_branch_size, left_branch_size);
      int min = std::min(right_branch_size, left_branch_size);

      data_type *old_buffer =
          min == left_branch_size ? left_branch_buffer : right_branch_buffer;

      data_type *temp = new data_type[max * n_components];
      std::memcpy(temp, old_buffer, min * n_components * sizeof(data_type));
      for (int i = min * n_components; i < max * n_components; ++i) {
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

    merging_array = new data_type[(branch_size * 2 + 1) * n_components];

    // the root of this tree is the data point used to split left and right
    split_item.copy_to_array(merging_array, n_components);

    rearrange_branches(merging_array + n_components, left_branch_buffer,
                       right_branch_buffer, branch_size, n_components);

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

    MPI_Send(left_branch_buffer, left_branch_size * n_components, mpi_data_type,
             parent, TAG_RIGHT_PROCESS_PROCESSING_OVER, MPI_COMM_WORLD);
  }
  *kdtree_size = left_branch_size;
  return left_branch_buffer;
}
