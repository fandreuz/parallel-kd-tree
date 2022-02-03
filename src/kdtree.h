#pragma once

#include "data_point.h"
#include "knode.h"
#include "process_utils.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <math.h>
#include <optional>
#include <tuple>
#include <unistd.h>
#include <vector>

#include <mpi.h>
#include <omp.h>

// a tuple which packs some info about the result of parallelization with MPI,
// which are then used in the parallelization with OpenMP.
using mpi_parallelization_result =
    std::tuple<std::vector<DataPoint>::iterator,
               std::vector<DataPoint>::iterator, int>;

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
 * @brief Communicate to a child process the data points some info regarding the
 *        branch it is assigned to.
 *
 * Attached to this communication there should be some info regarding the branch
 * (number of data points, depth  of the tree at this point, rank of the parent,
 * number of components in the data points).
 */
#define TAG_RIGHT_PROCESS_START_INFO 12
/**
 * @def
 * @brief Communicate to a child process the data points in the branch it is
 *          assigned to.
 *
 * Attached to this communication there should be an array containing the
 * data points assigned to this process (n_components consecutive values are
 * considered a data point).
 */
#define TAG_RIGHT_PROCESS_START_DATA 13

class KDTreeGreenhouse {
private:
  array_size n_datapoints;
  // number of data points assigned to this greenhouse, may decrease if a
  // portion is assigned to another MPI process.
  int n_components;

  // the depth which this k-d tree starts from when this MPI process is awaken,
  // used to determine which child process are to be used in further parallel
  // splittings.
  int starting_depth = 0;

  // rank of this MPI process
  int mpi_rank = -1;
  // children of this process, i.e. processes that received a right branch from
  // this MPI process.
  std::vector<int> children;
  // MPI rank of the parent process of this process
  int parent = -1;

  // DataPoint used to split a branch assigned to this MPI process. this process
  // then received the left branch resulting from the split.
  std::vector<DataPoint> parallel_splits;

  // an MPI_Communicator which does not hold the main process.
  MPI_Comm no_main_communicator;

  // number of MPI processes available
  int n_mpi_workers = -1;
  // number of OpenMP processes available
  int n_omp_workers = -1;

  // maximum depth of the tree at which we can parallelize using MPI. after this
  // depth no more right-branches can be assigned to non-surplus processes
  int max_mpi_depth = 0;
  // maximum depth of the tree at which we can parallelize using OpenMP. this
  // is used in practice to determine the maximum number of OpenMP tasks.
  int max_omp_depth = 0;

  // number of additional MPI processes that are not enough to parallelize an
  // entire level of the tree: they are assigned left-to-right to the processes
  // in the last fully-parallelizable level.
  int surplus_mpi_processes = 0;
  // number of additional OpenMP processes that are not enough to parallelize an
  // entire level of the tree. they are used in practice to determine the
  // maximum number of OpenMP tasks.
  int surplus_omp_processes = 0;

  // number of elements in the tree produced by this core. it's equal to the
  // smallest powersum of 2 bigger than the number of datapoints assigned to
  // this core.
  array_size tree_size = 0;
  // array of DataPoints already placed into the tree using OpenMP or serially.
  std::optional<DataPoint> *pending_tree = nullptr;

  // a pool of ready-to-use memory which can be used as a buffer to store
  // temporary the content of the right branch before sending it to the
  // appropriate MPI process.
  data_type *right_branch_memory_pool = nullptr;
  // an MPI request which monitors the operation of sending data to the right
  // branch.
  MPI_Request right_branch_send_data_request = MPI_REQUEST_NULL;

  // wait for another MPI process to pass the data assigned to this MPI process
  data_type *retrieve_dataset_info();

  mpi_parallelization_result
  start_mpi_growth(std::vector<DataPoint> &data_points);
  void start_omp_growth(mpi_parallelization_result mpi_result);

  array_size grown_kdtree_size = 0;
  data_type *grown_kdtree_1d = nullptr;
  KNode<data_type> *grown_kd_tree = nullptr;

  void grow_kd_tree(std::vector<DataPoint> &data_points);

  mpi_parallelization_result
  build_tree_mpi(std::vector<DataPoint>::iterator first_data_point,
                 std::vector<DataPoint>::iterator end_data_point, int depth);

  void build_tree_single_core(std::vector<DataPoint>::iterator first_data_point,
                              std::vector<DataPoint>::iterator end_data_point,
                              int depth, array_size region_width,
                              array_size region_start_index,
                              array_size branch_starting_index);

  void finalize_mpi();
  void finalize_single_core();

public:
  KDTreeGreenhouse(data_type *data, array_size n_datapoints, int n_components);
  ~KDTreeGreenhouse() { delete grown_kd_tree; }

  KNode<data_type> &&extract_grown_kdtree() {
    return std::move(*grown_kd_tree);
  }
  array_size get_grown_kdtree_size() { return grown_kdtree_size; }
};
