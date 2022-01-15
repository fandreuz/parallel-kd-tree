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
 * @brief Communicate to a child process the data points in the branch it is
 *          assigned to. Attached to this communication there should be some
 *          info regarding the branch (number of data points, depth of the tree
 *          at this point, rank of the parent, number of components in the
 *          data points).
 */
#define TAG_RIGHT_PROCESS_START 12

class KDTreeGreenhouse {
private:
  array_size n_datapoints;
  int n_components;

  // the depth which this k-d tree starts from, used to determine which
  // child process are to be used in further parallel splittings
  int starting_depth = 0;

  // rank of this process
  int rank = -1;

  // number of MPI processes available
  int n_parallel_workers = -1;

  // maximum depth of the tree at which we can parallelize. after this depth no
  // more right-branches can be assigned to non-surplus processes
  int max_depth = 0;

  // number of additional processes that are not enough to parallelize an entire
  // level of the tree, they are assigned left-to-right until there are no more
  // surplus processes
  int surplus_processes = 0;

  // number of items assigned serially (i.e. non-parallelizable) to this process
  array_size serial_tree_size = 0;
  // maximum depth of the serial sub-tree.
  int max_serial_depth = 0;

  // DataPoints in the serial branch assigned to this process. see also
  // build_tree_serial
  std::optional<DataPoint> *serial_tree = nullptr;

#ifdef USE_MPI
  // rank of the parent process of this process
  int parent = -1;

  // DataPoint used to split a branch assigned to this process. this process
  // then received the left branch resulting from the split.
  std::vector<DataPoint> parallel_splits;

  // children of this process, i.e. processes that received a right branch from
  // this process
  std::vector<int> children;
#endif

  array_size grown_kdtree_size;
  KNode<data_type> *grown_kd_tree = nullptr;

  data_type *grow_kd_tree(std::vector<DataPoint> data_points);

#ifdef USE_MPI
  data_type *retrieve_dataset_info();
  void build_tree_parallel(std::vector<DataPoint>::iterator first_data_point,
                           std::vector<DataPoint>::iterator end_data_point,
                           int depth);
#endif

  void build_tree_serial(std::vector<DataPoint>::iterator first_data_point,
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
