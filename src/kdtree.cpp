#include "kdtree.h"

KDTreeGreenhouse::KDTreeGreenhouse(data_type *data, array_size n_datapoints,
                                   int n_components)
    : n_datapoints{n_datapoints}, n_components{n_components} {
  bool should_delete_data = false;

// we are not interested in the rank for OpenMP
#ifdef USE_MPI
  rank = get_rank();
#endif

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

data_type *KDTreeGreenhouse::grow_kd_tree(std::vector<DataPoint> data_points) {
#pragma omp parallel
  {
#pragma omp single
    {
      n_parallel_workers = get_n_parallel_workers();
      max_parallel_depth = compute_max_depth(n_parallel_workers);
      surplus_workers =
          compute_n_surplus_processes(n_parallel_workers, max_parallel_depth);

#ifdef USE_MPI
      // we initialize the pool using the biggest possible number of components
      // we will ever need for this branch.
      right_branch_memory_pool = new data_type[n_datapoints / 2 * n_components];

      build_tree_parallel(data_points.begin(), data_points.end(),
                          starting_depth);

      // before deleting the pool, we wait the last communication to be
      // completed
      MPI_Wait(&right_branch_send_data_request, MPI_STATUS_IGNORE);
      delete[] right_branch_memory_pool;
#else
      // we want to store the tree (in a temporary way) in an array whose size
      // is a powersum of two
      tree_size = powersum_of_two(n_datapoints, true);
      growing_tree = new std::optional<DataPoint>[tree_size];

      int starting_region_width;
#ifdef ALTERNATIVE_SERIAL_WRITE
      starting_region_width = tree_size;
#else
      starting_region_width = 1;
#endif

      build_tree_single_core(data_points.begin(), data_points.end(), 0,
                             starting_region_width, 0, 0);
#endif
    }
  }

  if (!data_points.empty())
    return finalize();
  else
    return nullptr;
}
