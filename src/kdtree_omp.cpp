#include "kdtree.h"

/*
   Construct a tree on a single core (maybe via OpenMP). The current process
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
void KDTreeGreenhouse::build_tree_single_core(
    std::vector<DataPoint>::iterator first_data_point,
    std::vector<DataPoint>::iterator end_data_point, int depth,
    array_size region_width, array_size region_start_index,
    array_size branch_starting_index) {
  // this is equivalent to say that there is at most one data point in the
  // sequence
  if (first_data_point + 1 == end_data_point) {
    // if we encounter the flag ALTERNATIVE_SERIAL_WRITE the parameter
    // branch_starting_index will always be zero, therefore it does not
    // interphere with this writing.
    growing_tree[region_start_index + branch_starting_index].emplace(
        DataPoint(std::move(*first_data_point)));
  } else {
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
#ifndef USE_MPI
    bool no_spawn_more_threads =
        depth > max_parallel_depth + 1 ||
        (depth == max_parallel_depth + 1 && get_rank() >= surplus_workers);
#endif

    std::vector<DataPoint>::iterator right_branch_first_point =
        first_data_point + split_point_idx + 1;
#pragma omp task default(shared) final(no_spawn_more_threads)
    {
#ifdef DEBUG
      std::cout << "Task assigned to thread " << omp_get_thread_num()
                << std::endl;
#endif
      // right
      build_tree_single_core(right_branch_first_point, end_data_point, depth,
                             region_width, region_start_index_right,
                             branch_start_index_right);
    }
    // left
    if (split_point_idx > 0)
      build_tree_single_core(first_data_point, right_branch_first_point - 1,
                             depth, region_width, region_start_index_left,
                             branch_start_index_left);

// there are variables on the stack, we should wait before letting this
// function die. this is not a big deal since all recursive call are going to
// be there for a long time anyway.
#pragma omp taskwait
  }
}

data_type *KDTreeGreenhouse::finalize() {
  grown_kdtree_size = tree_size;
  data_type *tree = unpack_optional_array(growing_tree, tree_size, n_components,
                                          EMPTY_PLACEHOLDER);
#ifdef ALTERNATIVE_SERIAL_WRITE
  data_type *temp_tree = new data_type[tree_size * n_components];
  rearrange_kd_tree(temp_tree, tree, tree_size, n_components);
  delete[] tree;
  tree = temp_tree;
#endif
  return tree;
}
