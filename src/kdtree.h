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
#include <unistd.h>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#else
#include <omp.h>
#endif

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
  // number of datapoints managed by this class (might vary if some portions are
  // delegated to other processes).
  array_size n_datapoints;
  // number of components for each datapoint.
  int n_components;

  // the depth in the whole tree that this kd-tree is located at. used to
  // determine which child process are to be used in further parallel
  // splittings.
  int starting_depth = 0;

#ifdef USE_MPI
  // rank of this process
  int rank = -1;

  // an MPI_Communicator which does not hold the main process.
  MPI_Comm no_main_communicator;
#endif

  // number of MPI processes/OMP threads available
  int n_parallel_workers = -1;

  // maximum depth of the tree at which we can parallelize using OMP/MPI. after
  // this depth no more right-branches can be assigned to non-surplus
  // workers.
  int max_parallel_depth = 0;

  // number of additional processes/threads that are not enough to parallelize
  // an entire level of the tree, and are therefore assigned
  // left-to-right until there are no more left.
  int surplus_workers = 0;

  // number of items assigned to a particular core (i.e. non-parallelizable with
  // MPI).
  array_size tree_size = 0;

  // DataPoints assigned to this process/core. see also
  // build_tree_single_core().
  std::optional<DataPoint> *growing_tree = nullptr;

#ifdef USE_MPI
  // rank of the MPI parent process of this process
  int parent = -1;

  // DataPoint used to split a branch assigned to this process. this process
  // then received the left branch resulting from the split, and assigned the
  // right branch to another MPI process.
  std::vector<DataPoint> parallel_splits;

  // MPI children of this process, i.e. processes that received a right branch
  // from this process.
  std::vector<int> children;

  // a pool of ready-to-use memory which can be used as a buffer to store
  // temporary the content of the right branch before sending it to the
  // appropriate process.
  data_type *right_branch_memory_pool = nullptr;
  // an MPI request which monitors the operation of sending data in the right
  // branch to another MPI process.
  MPI_Request right_branch_send_data_request = MPI_REQUEST_NULL;
#endif

  // full size (different than the number of datapoints) of the kd-tree grown by
  // this greenhouse.
  array_size grown_kdtree_size = 0;
  // root node of the kd-tree grown by this greenhouse.
  KNode<data_type> *grown_kd_tree = nullptr;

  data_type *grow_kd_tree(std::vector<DataPoint> &data_points);

#ifdef USE_MPI
  data_type *retrieve_dataset_info();
  void build_tree_parallel(std::vector<DataPoint>::iterator first_data_point,
                           std::vector<DataPoint>::iterator end_data_point,
                           int depth);
#endif

  void build_tree_single_core(std::vector<DataPoint>::iterator first_data_point,
                              std::vector<DataPoint>::iterator end_data_point,
                              int depth, array_size region_width,
                              array_size region_start_index,
                              array_size branch_starting_index);
  data_type *finalize();

public:
  KDTreeGreenhouse(data_type *data, array_size n_datapoints, int n_components);
  ~KDTreeGreenhouse() { delete grown_kd_tree; }

  KNode<data_type> &&extract_grown_kdtree() {
    return std::move(*grown_kd_tree);
  }
  array_size get_grown_kdtree_size() { return grown_kdtree_size; }
};
