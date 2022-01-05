#include "data_point.h"
#include "tree.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
#include <vector>

/**
 * @brief Generate a k-d tree a set of k-dimensional data.
 *
 * Generate a k-d tree from the given `data`.
 *
 * If this process is not the main MPI process (i.e. it does not rank 0), the
 * process freezes until another thread sends some work to perform. The control
 * is then returned to the calling function as soon as the work assigned to
 * this process is done, or some other process notifies that this process is not
 * going to be used in the construction of the tree.
 *
 * @param data 1D array of data, `k` consecutive items constitute a data point
 *              in the dataset.
 * @param size Dimension of the dataset, i.e. `length(data) / dms`.
 * @param dms  Number of components that constitute a data point in the dataset.
 * @return The root of the tree (i.e. the first data point used to split the
 *          dataset).
 */
KNode<data_type> *generate_kd_tree(data_type *data, int size, int dms);
