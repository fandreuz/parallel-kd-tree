#include "kdtree.h"

KDTreeGreenhouse::KDTreeGreenhouse(data_type *data, array_size n_datapoints,
                                   int n_components)
    : n_datapoints{n_datapoints}, n_components{n_components} {
  bool should_delete_data = false;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (data == nullptr) {
    should_delete_data = true;
    data = retrieve_dataset_info();
  }

  std::vector<DataPoint> data_points =
      as_data_points(data, this->n_datapoints, this->n_components);
  // 1D representation of our KDTree, or nullptr (if not main process)
  data_type *tree = grow_kd_tree(data_points);

  // tree is nullptr if the branch assigned to this process (in MPI) was empty
  if (tree != nullptr)
    grown_kd_tree =
        convert_to_knodes(tree, grown_kdtree_size, this->n_components, 0, 1, 0);
  else {
    grown_kd_tree = new KNode<data_type>();
  }

  if (should_delete_data)
    delete[] data;
}

data_type *KDTreeGreenhouse::grow_kd_tree(std::vector<DataPoint> &data_points) {
  MPI_Comm_size(MPI_COMM_WORLD, &n_mpi_processes);
  max_mpi_parallel_depth = compute_max_depth(n_mpi_processes);
  surplus_mpi_workers =
      compute_n_surplus_processes(n_mpi_processes, max_mpi_parallel_depth);

#ifdef DEBUG
  std::cout << "Starting parallel MPI region with " << max_mpi_parallel_depth
            << " parallel workers." << std::endl;
#endif

  // we initialize the pool using the biggest possible number of components
  // we will ever need for this branch.
  right_branch_memory_pool = new data_type[n_datapoints / 2 * n_components];

  build_tree_mpi(data_points.begin(), data_points.end(), starting_depth);

  // before deleting the pool, we wait the last communication to be
  // completed
  MPI_Wait(&right_branch_send_data_request, MPI_STATUS_IGNORE);
  delete[] right_branch_memory_pool;

  if (!data_points.empty()) {
    data_type *single_process_tree = finalize_single_process();

    data_type *tree = finalize_mpi(single_process_tree);

    // we do not need to free this because:
    // 1. if tree == single_process_tree we actually have not to free
    // 2. otherwise, tree is single_process_tree is deleted inside finalize_mpi
    //if (tree != single_process_tree)
    //  delete[] single_process_tree;

    return tree;
  } else
    return nullptr;
}

/*
   Construct a tree on a single process (maybe via OpenMP). The current process
   takes care of both the left and right branch.

   A region (i.e. k contiguous elements) of growing_tree holds an entire level
   of the k-d tree (i.e. elements whose distance from the root is the same).

   - first_data_point is an iterator pointing to the first data point in the
      set;
   - end_data_point is an iterator pointing to past-the-last data point;
   - depth is the depth of the tree after the addition of this new level;
   - region_width is the width of the current region of growing_tree which
      holds the current level of the tree. this increases (multiplied
      by 2) at each recursive call;
   - region_start_index is the index of growing_tree in which the region
      corresponding to the current level starts (i.e. the index after the end
      of the region corresponding to the level before);
   - branch_starting_index is the index of growing_tree (starting from
      region_width) in which the item used to split this branch is stored.
*/
void KDTreeGreenhouse::build_tree_single_process(
    std::vector<DataPoint>::iterator first_data_point,
    std::vector<DataPoint>::iterator end_data_point, int depth,
    array_size region_width, array_size region_start_index,
    array_size branch_starting_index) {
  // this is equivalent to say that there is at most one data point in the
  // sequence
  if (first_data_point + 1 != end_data_point) {
    int dimension = select_splitting_dimension(depth, n_components);
    array_size split_point_idx =
        sort_and_split(first_data_point, end_data_point, dimension);

    growing_tree[region_start_index + branch_starting_index].emplace(
        DataPoint(std::move(*(first_data_point + split_point_idx))));

    array_size region_start_index_left, region_start_index_right;
    array_size branch_start_index_left, branch_start_index_right;

    // we update the values for the next iteration
#ifdef ALTERNATIVE_SERIAL_WRITE
    // the two processes which handle the halves are going to have less space
    // at their disposal
    region_width = (region_width - 1) / 2;
    // in this case we divide the available space in two halves and assign each
    // half to the two processes/threads
    region_start_index_left = region_start_index + 1;
    region_start_index_right = region_start_index_left + region_width;

    branch_start_index_left = branch_start_index_right = 0;
#else
    region_start_index_left = region_start_index_right =
        region_start_index + region_width;

    // the width of the next level will be twice the width of the current
    // level
    region_width *= 2;

    branch_starting_index *= 2;
    branch_start_index_left = branch_starting_index;
    branch_start_index_right = branch_starting_index + 1;
#endif
    depth += 1;

    // in case we're on OpenMP, we need to understand whether we can spawn more
    // OpenMP threads
    bool no_spawn_more_threads = depth > max_omp_parallel_depth + 1 ||
                                 (depth == max_omp_parallel_depth + 1 &&
                                  omp_get_thread_num() >= surplus_omp_workers);

    std::vector<DataPoint>::iterator right_branch_first_point =
        first_data_point + split_point_idx + 1;
#pragma omp task final(no_spawn_more_threads)
    {
#ifdef DEBUG
      std::cout << "Task assigned to thread " << omp_get_thread_num()
                << std::endl;
#endif
      // right
      build_tree_single_process(right_branch_first_point, end_data_point, depth,
                                region_width, region_start_index_right,
                                branch_start_index_right);
    }
    // left
    if (split_point_idx > 0)
      build_tree_single_process(first_data_point, right_branch_first_point - 1,
                                depth, region_width, region_start_index_left,
                                branch_start_index_left);

// there are variables on the stack, we should wait before letting this
// function die. this is not a big deal since all recursive call are going to
// be there for a long time anyway.
#pragma omp taskwait
  } else {
    // if we encounter the flag ALTERNATIVE_SERIAL_WRITE the parameter
    // branch_starting_index will always be zero, therefore it does not
    // interphere with this writing.
    growing_tree[region_start_index + branch_starting_index].emplace(
        DataPoint(std::move(*first_data_point)));
  }
}

data_type *KDTreeGreenhouse::finalize_single_process() {
  grown_single_process_kdtree_size = tree_size;
  data_type *tree =
      unpack_optional_array(growing_tree, grown_single_process_kdtree_size,
                            n_components, EMPTY_PLACEHOLDER);
#ifdef ALTERNATIVE_SERIAL_WRITE
  data_type *temp_tree =
      new data_type[grown_single_process_kdtree_size * n_components];
  rearrange_kd_tree(temp_tree, tree, grown_single_process_kdtree_size,
                    n_components);
  delete[] tree;
  tree = temp_tree;
#endif
  return tree;
}
