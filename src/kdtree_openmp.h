#include "process_utils.h"
#include "utils.h"

#include <omp.h>

/**
 * @brief Generate a k-d tree a set of k-dimensional data using OpenMP.
 *
 * Generate a k-d tree from the given `data` using OpenMP.
 *
 * @param data 1D array of data, `k` consecutive items constitute a data point
 *              in the dataset.
 * @param size Dimension of the dataset, i.e. `length(data) / dms`.
 * @param dms  Number of components that constitute a data point in the dataset.
 * @return The root of the tree (i.e. the first data point used to split the
 *          dataset).
 */
KNode<data_type> *generate_kd_tree(data_type *data, int size, int dms);
