#include "kdtree.h"

KDTreeGreenhouse::KDTreeGreenhouse(data_type *data, int n_datapoints,
                                   int n_components)
    : n_datapoints{n_datapoints}, n_components{n_components} {
  bool should_delete_data = false;
  if (data == nullptr) {
#ifdef USE_MPI
    should_delete_data = true;
    data = retrieve_dataset_info();
#else
    throw std::invalid_argument("Received a null dataset.");
#endif
  }

  // 1D representation of our KDTree, or nullptr (if not main process)
  data_type *tree = grow_kd_tree(
      as_data_points(data, this->n_datapoints, this->n_components));
  grown_kd_tree =
      convert_to_knodes(tree, grown_kdtree_size, this->n_components, 0, 1, 0);

  if (should_delete_data)
    delete[] data;
}

data_type *KDTreeGreenhouse::grow_kd_tree(std::vector<DataPoint> data_points) {
#pragma omp parallel
  {
#pragma omp single
    {
      n_parallel_workers = get_n_parallel_workers();
      max_depth = compute_max_depth(n_parallel_workers);
      surplus_processes =
          compute_n_surplus_processes(n_parallel_workers, max_depth);
      rank = get_rank();

#ifdef USE_MPI
      build_tree_parallel(data_points.begin(), data_points.end(),
                          starting_depth);
#else
      // we want to store the tree (in a temporary way) in an array whose size
      // is a powersum of two
      serial_tree_size = bigger_powersum_of_two(n_datapoints);
      serial_tree = new std::optional<DataPoint>[serial_tree_size];

      build_tree_serial(data_points.begin(), data_points.end(), 0, 1, 0, 0);
#endif
    }
  }

  return finalize();
}

/*
   Construct a tree serially. The current process takes care of both the left
   and right branch.

   A region (i.e. k contiguous elements) of serial_tree holds an entire level
   of the k-d tree (i.e. elements whose distance from the root is the same).

   - first_data_point is an iterator pointing to the first data point in the
      set;
   - end_data_point is an iterator pointing to past-the-last data point;
   - depth is the depth of the tree after the addition of this new level;
   - region_width is the width of the current region of serial_tree which
      holds the current level of the serial tree. this increases (multiplied
      by 2) at each recursive call;
   - region_start_index is the index of serial_tree in which the region
      corresponding to the current level starts (i.e. the index after the end
      of the region corresponding to the level before);
   - branch_starting_index is the index of serial_tree (starting from
      region_width) in which the item used to split this branch is stored.
*/
void KDTreeGreenhouse::build_tree_serial(
    std::vector<DataPoint>::iterator first_data_point,
    std::vector<DataPoint>::iterator end_data_point, int depth,
    int region_width, int region_start_index, int branch_starting_index) {
  // this is equivalent to say that there is at most one data point in the
  // sequence
  if (first_data_point == end_data_point ||
      first_data_point + 1 == end_data_point) {
#ifdef DEBUG
    std::cout << "[rank" << rank << "]: hit the bottom! " << std::endl;
#endif
    serial_tree[region_start_index + branch_starting_index].emplace(
        DataPoint(std::move(*first_data_point)));
  } else {
    int dimension = select_splitting_dimension(depth, n_components);
    int split_point_idx =
        sort_and_split(first_data_point, end_data_point, dimension);

#ifdef DEBUG
    std::cout << "[rank" << rank << "]: serial split against axis " << dimension
              << ", split_idx = " << split_point_idx
              << ", size = " << std::distance(first_data_point, end_data_point)
              << std::endl;
#endif

    serial_tree[region_start_index + branch_starting_index].emplace(
        DataPoint(std::move(*(first_data_point + split_point_idx))));

    // we update the values for the next iteration
    region_start_index += region_width;
    region_width *= 2;
    branch_starting_index *= 2;
    depth += 1;

// in case we're on OpenMP, we need to understand whether we can spawn more
// OpenMP threads
#ifndef USE_MPI
    bool no_spawn_more_threads =
        depth > max_depth + 1 ||
        (depth == max_depth + 1 && rank >= surplus_processes);
#endif

    std::vector<DataPoint>::iterator right_branch_first_point =
        first_data_point + split_point_idx + 1;
#pragma omp task default(shared) final(no_spawn_more_threads)
    {
      // right
      build_tree_serial(right_branch_first_point, end_data_point, depth,
                        region_width, region_start_index,
                        branch_starting_index + 1);
    }
    // left
    if (split_point_idx > 0)
      build_tree_serial(first_data_point, right_branch_first_point - 1, depth,
                        region_width, region_start_index,
                        branch_starting_index);

// there are variables on the stack, we should wait before letting this
// function die. this is not a big deal since all recursive call are going to
// be there for a long time anyway.
#pragma omp taskwait
  }
}
