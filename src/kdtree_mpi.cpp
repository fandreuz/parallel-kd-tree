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
           TAG_RIGHT_PROCESS_START_INFO, MPI_COMM_WORLD, &status);

  // number of data points in the branch
  n_datapoints = br_size_depth_parent[0];
  // depth of the tree at this point
  starting_depth = br_size_depth_parent[1];
  // rank of the parent which "started" (i.e. waked) this process
  parent = br_size_depth_parent[2];
  n_components = br_size_depth_parent[3];

  data_type *data = nullptr;
  if (n_datapoints > 0) {
    data = new data_type[n_datapoints * n_components];
    // receive the data in the branch assigned to this process
    MPI_Recv(data, n_datapoints * n_components, mpi_data_type, MPI_ANY_SOURCE,
             TAG_RIGHT_PROCESS_START_DATA, MPI_COMM_WORLD, &status);
  }

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
void KDTreeGreenhouse::build_tree_parallel(
    std::vector<DataPoint>::iterator first_data_point,
    std::vector<DataPoint>::iterator end_data_point, int depth) {
#ifdef MPI_DEBUG
  if (depth == starting_depth) {
    int debug_rank = atoi(getenv("MPI_DEBUG_RANK"));
    std::cerr << "MPI_DEBUG_RANK=" << atoi(getenv("MPI_DEBUG_RANK"))
              << std::endl;
    if (rank == debug_rank) {
      volatile int i = 0;
      char hostname[256];
      gethostname(hostname, sizeof(hostname));
      printf("PID %d on %s ready for attach\n", getpid(), hostname);
      fflush(stdout);
      while (0 == i)
        sleep(5);
    }
  }
#endif

  int next_depth = depth + 1;
  int right_process_rank =
      compute_next_process_rank(rank, max_parallel_depth, next_depth,
                                surplus_workers, n_parallel_workers);

  if (right_process_rank == -1) {
    if (n_datapoints > 0) {
#ifdef DEBUG
      std::cout << "[rank" << rank
                << "]: no available processes, going single core from now "
                << std::endl;
#endif

      // we want that the single core branch is storable in an array whose size
      // is a powersum of two
      tree_size = powersum_of_two(n_datapoints, true);
      // sum of them are NOT going to be initialized since they are placeholders
      // of leafs (last level of the tree) that are not present since
      // n_datapoints < powersum_of_two
      growing_tree = new std::optional<DataPoint>[tree_size];

      int starting_region_width;
#ifdef ALTERNATIVE_SERIAL_WRITE
      starting_region_width = tree_size;
#else
      starting_region_width = 1;
#endif

      build_tree_single_core(first_data_point, end_data_point, depth,
                             starting_region_width, 0, 0);
    } else {
#ifdef DEBUG
      std::cout << "[rank" << rank << "]: build_tree_parallel is dead now "
                << std::endl;
#endif
      // otherwise there's nothing to do
    }
  } else {
    int dimension = select_splitting_dimension(depth, n_components);
    array_size split_point_idx = 0;
    array_size right_branch_size = 0;
    if (n_datapoints > 0) {
      split_point_idx =
          sort_and_split(first_data_point, end_data_point, dimension);
      parallel_splits.push_back(
          std::move(*(first_data_point + split_point_idx)));
      right_branch_size = n_datapoints - split_point_idx - 1;

      // even if we did not send anything to the right process, we still need
      // to store that process in children because otherwise we cannot recover
      // the corresponding datapoint from parallel_splits.
      children.push_back(right_branch_size > 0 ? right_process_rank : -1);

#ifdef DEBUG
      std::cout << "[rank" << rank << "]: parallel split against axis "
                << dimension << ", split_idx = " << split_point_idx
                << ", rank of the expected children is "
                << children[children.size() - 1] << std::endl;
#endif
    } else {
#ifdef DEBUG
      std::cout << "[rank" << rank
                << "]: simulating a parallel split to wake rank"
                << right_process_rank << std::endl;
#endif
    }

#ifdef DEBUG
    std::cout << "[rank" << rank
              << "]: delegating right region (starting from) "
              << split_point_idx + 1 << " (size " << right_branch_size << " of "
              << n_datapoints << ") to rank" << right_process_rank << std::endl;
#endif
    MPI_Request info_request;

    int right_branch_data[4];
    right_branch_data[0] = right_branch_size;
    right_branch_data[1] = next_depth;
    right_branch_data[2] = rank;
    right_branch_data[3] = n_components;
    MPI_Isend(right_branch_data, 4, MPI_INT, right_process_rank,
              TAG_RIGHT_PROCESS_START_INFO, MPI_COMM_WORLD, &info_request);

    std::vector<DataPoint>::iterator right_branch_first_point =
        first_data_point + split_point_idx + 1;

    if (right_branch_size > 0) {
      // before overwriting the right branch pool we wait to see if the last
      // comunication is completed. otherwise we may corrupt the buffer
      MPI_Wait(&right_branch_send_data_request, MPI_STATUS_IGNORE);
      // we unpack the right branch into the right pool
      unpack_array(right_branch_memory_pool, right_branch_first_point,
                   end_data_point, n_components);

      // we delegate the right part to another process
      // this is synchronous since we also want to delete the buffer ASAP
      MPI_Isend(right_branch_memory_pool, right_branch_size * n_components,
                mpi_data_type, right_process_rank, TAG_RIGHT_PROCESS_START_DATA,
                MPI_COMM_WORLD, &right_branch_send_data_request);
    }

    n_datapoints = split_point_idx;

    // this process takes care of the left part
    build_tree_parallel(first_data_point, right_branch_first_point - 1,
                        next_depth);

    // the buffer will be destroyed when the stack is destroyed, therefore we
    // wait the end of the asynchronous operation
    MPI_Wait(&info_request, MPI_STATUS_IGNORE);
  }
}

data_type *KDTreeGreenhouse::finalize() {
  // we wait for all the child processes to complete their work
  int n_children = children.size();

#ifdef DEBUG
  std::cout << "[rank" << rank
            << "]: finalize called. #children = " << n_children << std::endl;
#endif

  int right_rank = -1;
  array_size right_branch_size = 0;
  // buffer which contains the split indexes from the right branch
  data_type *right_branch_buffer = nullptr;

  data_type *left_branch_buffer = nullptr;
  array_size left_branch_size = tree_size;

  if (tree_size > 0) {
    left_branch_buffer = unpack_optional_array(growing_tree, tree_size,
                                               n_components, EMPTY_PLACEHOLDER);
#ifdef ALTERNATIVE_SERIAL_WRITE
    data_type *temp_left_buffer = new data_type[tree_size * n_components];
    rearrange_kd_tree(temp_left_buffer, left_branch_buffer, tree_size,
                      n_components);
    delete[] left_branch_buffer;
    left_branch_buffer = temp_left_buffer;
#endif
  }

  // merged_array contains the values which results from merging a right branch
  // with a left branch.
  data_type *merging_array;
  for (int i = n_children - 1; i >= 0; --i) {
    right_rank = children.at(i);

    // if right_rank is -1 we are assuming that the right branch given to the
    // child project was a "placeholder", therefore we do not need to contact
    // the child.
    if (right_rank != -1) {
      MPI_Recv(&right_branch_size, 1, MPI_INT, right_rank,
               TAG_RIGHT_PROCESS_N_ITEMS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // TODO: this can be optimized
    right_branch_buffer = new data_type[right_branch_size * n_components];

    if (right_rank != -1) {
      // we gather the branch from another process
      MPI_Recv(right_branch_buffer, right_branch_size * n_components,
               mpi_data_type, right_rank, TAG_RIGHT_PROCESS_PROCESSING_OVER,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    array_size branch_size = left_branch_size;
    if (right_branch_size != left_branch_size) {
      array_size max = std::max(right_branch_size, left_branch_size);
      array_size min = std::min(right_branch_size, left_branch_size);

      data_type *old_buffer =
          min == left_branch_size ? left_branch_buffer : right_branch_buffer;

      data_type *temp = new data_type[max * n_components];
      std::memcpy(temp, old_buffer, min * n_components * sizeof(data_type));
      for (array_size i = min * n_components; i < max * n_components; ++i) {
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

    DataPoint split_item = std::move(parallel_splits[i]);

    merging_array = new data_type[(branch_size * 2 + 1) * n_components];

    // the root of this tree is the data point used to split left and right
    split_item.copy_to_array(merging_array, n_components);

    merge_kd_trees(merging_array + n_components, left_branch_buffer,
                   right_branch_buffer, branch_size, n_components);

    delete[] right_branch_buffer;
    delete[] left_branch_buffer;

    // we go one level up, therefore the merging array is now the array that
    // represents the left branch buffer
    left_branch_buffer = merging_array;

    // the new size of the left branch is the sum of the former left branch size
    // and of the right branch size, plus 1 (the split point)
    left_branch_size = branch_size * 2 + 1;

    right_branch_size = 0;
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

  grown_kdtree_size = left_branch_size;
  return left_branch_buffer;
}
