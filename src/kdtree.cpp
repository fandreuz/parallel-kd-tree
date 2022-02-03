#include "kdtree.h"

KDTreeGreenhouse::KDTreeGreenhouse(data_type *data, array_size n_datapoints,
                                   int n_components)
    : n_datapoints{n_datapoints}, n_components{n_components} {
  // MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_mpi_workers);
  max_mpi_depth = compute_max_depth(n_mpi_workers);
  surplus_mpi_processes =
      compute_n_surplus_processes(n_mpi_workers, max_mpi_depth);

  // OpenMP
  n_omp_workers = omp_get_num_threads();
  max_omp_depth = compute_max_depth(n_omp_workers);
  surplus_omp_processes =
      compute_n_surplus_processes(n_omp_workers, max_omp_depth);

  // if this MPI process is not main, it's going to receive `nullptr` in the
  // parameter `data`. therefore we need to retrieve the dataset from a parent
  // MPI process.
  // in the end we need to delete that dataset in order not to leak it. we must
  // not delete the data if we're on the main MPI process because we do not
  // own the data.
  bool should_delete_data = false;
  if (data == nullptr) {
    should_delete_data = true;
    data = retrieve_dataset_info();
  }

  std::vector<DataPoint> data_points =
      as_data_points(data, this->n_datapoints, this->n_components);
  grow_kd_tree(data_points);

  if (should_delete_data)
    delete[] data;
}

void KDTreeGreenhouse::grow_kd_tree(std::vector<DataPoint> &data_points) {
  auto mpi_result = start_mpi_growth(data_points);

  // as soon as we are done with MPI we can start parallelizing using OpenMP,
  // but only if there is still something to process
  if (std::get<0>(mpi_result) != std::get<1>(mpi_result))
    start_omp_growth(mpi_result);

  finalize_single_core();
  finalize_mpi();

  // tree is nullptr if the branch assigned to this MPI process was empty
  if (!data_points.empty())
    grown_kd_tree = convert_to_knodes(grown_kdtree_1d, grown_kdtree_size,
                                      this->n_components, 0, 1, 0);
  else {
    grown_kd_tree = new KNode<data_type>();
  }
}

mpi_parallelization_result
KDTreeGreenhouse::start_mpi_growth(std::vector<DataPoint> &data_points) {
  // we initialize the pool using the biggest possible number of components
  // we will ever need for this branch.
  right_branch_memory_pool = new data_type[n_datapoints / 2 * n_components];
  // start parallel construction of the kd-tree using MPI.
  auto tp = build_tree_mpi(data_points.begin(), data_points.end(),
                                starting_depth);

  // before deleting the pool, we wait the last communication to be
  // completed
  MPI_Wait(&right_branch_send_data_request, MPI_STATUS_IGNORE);
  delete[] right_branch_memory_pool;

  return tp;
}

void KDTreeGreenhouse::start_omp_growth(mpi_parallelization_result mpi_result) {
#pragma omp parallel
  {
#pragma omp single
    {
      // we want to store the tree (in a temporary way) in an array whose size
      // is a powersum of two
      tree_size = powersum_of_two(n_datapoints, true);
      // some of them are NOT going to be initialized since they are
      // placeholders of leafs (last level of the tree) that are not present
      // since n_datapoints < powersum_of_two.
      pending_tree = new std::optional<DataPoint>[tree_size];

      array_size starting_region_width;
#ifdef ALTERNATIVE_SERIAL_WRITE
      starting_region_width = tree_size;
#else
      starting_region_width = 1;
#endif

      std::vector<DataPoint>::iterator first_data_point =
          std::get<0>(mpi_result);
      std::vector<DataPoint>::iterator end_data_point = std::get<1>(mpi_result);
      int depth = std::get<2>(mpi_result);

      build_tree_single_core(first_data_point, end_data_point, depth,
                             starting_region_width, 0, 0);
    }
  }
}
