#include "kdtree_openmp.h"

// number of components for each data point
int dims = -1;

// maximum depth of the tree at which we can parallelize. after this depth no
// more right-branches can be assigned to non-surplus processes
int max_depth = 0;

// number of OpenMP threads
int num_threads;

// list of DataPoints which represents the tree as a 1D array, saved recursively
// as follows:
// we take the first item of the array, and put there the DataPoint used to
// split the tree. the left branch is the first half of the remaining array
// after the first item, the right branch is the second half
DataPoint *splits_tree = nullptr;
// the size of splits_tree, always a powersum of 2
int splits_tree_size;

// if the i-th item is true, the i-th item in serial_splits is initialized
bool *initialized;

// number of additional processes that are not enough to parallelize an entire
// level of the tree, they are assigned left-to-right until there are no more
// surplus processes
int surplus_processes = 0;

void build_tree(DataPoint *array, int size, int depth, int region_width,
                int region_start_index, int branch_starting_index);
data_type *finalize();

KNode<data_type> *generate_kd_tree(data_type *data, int size, int dms) {
  // we can save dims as a global variable since it is not going to change. it
  // is also constant for all the processes.
  dims = dms;

  num_threads = omp_get_num_threads();

  max_depth = compute_max_depth(num_threads);
  surplus_processes = compute_n_surplus_processes(num_threads, max_depth);
#ifdef DEBUG
  std::cout << "Starting " << num_threads << " with max_depth = " << max_depth
            << std::endl;
#endif

  // we create an array which packs all the data in a convenient way.
  // this weird mechanic is needed because we do not want to call the default
  // constructor (which the plain 'new' does)
  DataPoint *array = (DataPoint *)::operator new(size * sizeof(DataPoint));
  for (int i = 0; i < size; i++) {
    new (array + i) DataPoint(data + i * dims, dims);
  }

  // we want to store the tree (in a temporary way) in an array whose size is
  // a powersum of two
  splits_tree_size = bigger_powersum_of_two(size);
  splits_tree =
      (DataPoint *)::operator new(splits_tree_size * sizeof(DataPoint));

  initialized = new bool[splits_tree_size];
  for (int i = 0; i < splits_tree_size; ++i) {
    initialized[i] = false;
  }

#pragma omp parallel
  build_tree(array, size, 0, splits_tree_size, 0, );

  // when we reach this point, all the children threads have done their work
  // due to taskwait.

  // size might be changed by finalize (the actual size of the tree may not
  // be equal to the original size of the dataset)
  data_type *tree = finalize();
  return convert_to_knodes(tree, splits_tree_size, dims, 0, 1, 0);
}

/*
   Construct a tree. The current thread takes care of the left
   brannch, another OpenMP thread (via task mechanism) is going to take on the
   right branch, until no more threads are available.

   A region (i.e. k contiguous elements) of serial_splits holds an entire level
   of the k-d tree (i.e. elements whose distance from the root is the same).

   - array is the set of values to be inserted into the tree.
   - size is the size of array.
   - depth is the depth of the tree after the addition of this new level.
   - region_width is the width of the current region of serial_splits which
      holds the current level of the serial tree. this increases (multiplied
      by 2) at each recursive call.
   - region_start_index is the index of serial_splits in which the region
      corresponding to the current level starts (i.e. the index after the end
      of the region corresponding to the level before).
   - branch_starting_index is the index of serial_splits (starting from
      region_width) in which the item used to split this branch is stored.
*/
void build_tree(DataPoint *array, int size, int depth, int region_width,
                int region_start_index, int branch_starting_index) {
  initialized[region_start_index + branch_starting_index] = true;

  if (size <= 1) {
#ifdef DEBUG
    std::cout << "[rank" << omp_get_thread_num() << "]: hit the bottom! "
              << std::endl;
#endif
    new (splits_tree + region_start_index + branch_starting_index)
        DataPoint(std::move(array[0]));
  } else {
    int dimension = select_splitting_dimension(depth, dims);
    int split_point_idx = sort_and_split(array, size, dimension);

#ifdef DEBUG
    std::cout << "[rank" << omp_get_thread_num() << "]: split against axis "
              << dimension << ", split_idx = " << split_point_idx
              << ", size = " << size << std::endl;
#endif

    new (splits_tree + region_start_index + branch_starting_index)
        DataPoint(std::move(array[split_point_idx]));

    // we update the values for the next iteration
    region_start_index += region_width;
    region_width *= 2;
    branch_starting_index *= 2;
    depth += 1;

    int thread_num = omp_get_thread_num();
    bool no_more_new_threads =
        next_depth > max_depth + 1 ||
        (next_depth == max_depth + 1 && thread_num >= surplus_processes);

    // right
#pragma omp task shared(splits_tree) final(no_more_new_threads)
    build_tree(array + split_point_idx + 1, depth, size - split_point_idx - 1,
               region_width, region_start_index, branch_starting_index + 1);
    // left
    if (split_point_idx > 0)
      build_tree(array, split_point_idx, depth, region_width,
                 region_start_index, branch_starting_index);

// there are variables on the stack, we should wait before letting this
// function die. this is not a big deal since all recursive call are going to
// be there for a long time anyway.
#pragma omp taskwait
  }
}

data_type *finalize() {
#ifdef DEBUG
  std::cout << "[rank" << omp_get_thread_num() << "]: finalize called"
            << std::endl;
#endif

  // we unpack here the
  data_type *temp_buffer =
      unpack_risky_array(splits_tree, splits_tree_size, dims, initialized);

  // here we store the rearranged tree
  data_type *buffer = new data_type[splits_tree_size];

  // the root of this tree is the first element in temp_buffer
  std::memcpy(buffer, temp_buffer, dims * sizeof(data_type));

  int branch_size = (splits_tree_size - 1) / 2;

  data_type *left_branch = temp_buffer + dims;
  data_type *right_branch = temp_buffer + (1 + branch_size) * dims;

  rearrange_branches(buffer + dims, left_branch, right_branch, branch_size,
                     dims);
  delete[] temp_buffer;

  return buffer;
}
