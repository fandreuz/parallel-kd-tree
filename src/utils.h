#pragma once

#include "data_point.h"
#include "knode.h"
#include "tree_printer.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

/**
 * @def
 * @brief A placeholder used to fill holes in the 1D array which represents the
 *          three (this is needed for indexing reasons).
 */
#define EMPTY_PLACEHOLDER std::numeric_limits<int>::min()

/**
 * @brief Return the powersum of 2 nearest to the integer n.
 *
 * Depending on the value of `greater`, this function returns the smallest
 * powersum of 2 such that `n` is smaller than the sum (`greater == true`), or
 * the biggest powersum of 2 such that `n` is greater than the sum
 * (`greater == false`).
 *
 * Example:
 *
 *     powersum_of_two(5, false) = 3 = 1 + 2
 *     powersum_of_two(7, false) = 7 = 1 + 2 + 4
 *     powersum_of_two(5, true) = 7 = 1 + 2 + 4
 *     powersum_of_two(3, true) = 3 = 1 + 2
 */
int powersum_of_two(array_size n, bool greater);

/**
 * @brief Transform the given sequence of data points in a 1D array such that
 *          `n_components` contiguous items constitute a data point.
 *
 * @param first_point Iterator pointing to the first item of the sequence.
 * @param last_point Iterator pointing past the last item of the sequence.
 * @param n_components  Number of components for each data point.
 * @return data_type* A 1D array of size `size*n_components`.
 */
data_type *unpack_array(std::vector<DataPoint>::iterator first_point,
                        std::vector<DataPoint>::iterator last_point,
                        int n_components);

/**
 * @brief Transform the given sequence of data points in a 1D array such that
 *          `n_components` contiguous items constitute a data point.
 *
 * @param dest The array in which the unpacked elements will be placed.
 * @param first_point Iterator pointing to the first item of the sequence.
 * @param last_point Iterator pointing past the last item of the sequence.
 * @param n_components  Number of components for each data point.
 * @return data_type* A 1D array of size `size*n_components`.
 */
void unpack_array(data_type *dest, std::vector<DataPoint>::iterator first_point,
                  std::vector<DataPoint>::iterator last_point,
                  int n_components);

/**
 * @brief Transform the given array of data points (which may contain
 *           uninitialized values) in a 1D array such that `n_components`
 *           contiguous items constitute a data point.
 *
 * @param array 1D array of data points.
 * @param n_datapoints  Number of data points in the array (i.e.
 * `length(array)`).
 * @param n_components  Number of components for each data point.
 * @param initialized A 1D boolean array (same size of `array`) whose i-th
 *                      element is `true` if and only if the i-th element of
 *                      `array` has been initialized.
 * @return data_type* A 1D array of size `size*n_components`.
 */
data_type *unpack_optional_array(std::optional<DataPoint> *array,
                                 array_size n_datapoints, int n_components,
                                 data_type fallback_value);

/**
 * @brief Rearrange the two k-d trees `branch1`, `branch2` into a single array.
 *
 * Rearranges `branch1` and `branch2` into `dest` in such a way that:
 *
 *  1. We first take 1 node from branch1 and 1 node from branch2,
 *  2. Then 2 nodes from `branch1`, and 2 nodes from `branch2`;
 *  3. Then 4 nodes from `branch1`, and 4 nodes from `branch2`;
 *  4. And so on..
 *
 * @param dest    1D array in which we are going to store the content of
 *                  `branch1`, `branch2`.
 * @param branch1 The first branch, from which we are going to take nodes first.
 * @param branch1 The second branch, from which we are going to take nodes after
 *                  the first.
 * @param branches_size Size of `branch1` and `branch2` (number of data points,
 *                        **not** number of data points times the number of
 *                        dimensions).
 * @param n_components    Number of dimensions for each data point.
 */
void merge_kd_trees(data_type *dest, data_type *branch1, data_type *branch2,
                    array_size branches_size, int n_components);

#ifdef ALTERNATIVE_SERIAL_WRITE
void rearrange_kd_tree(data_type *dest, data_type *src, array_size subtree_size,
                       int n_components);
#endif

/**
 * @brief Convert the given tree to a kind-of linked list structure. This
 * assumes that the given size is a powersum of two.
 *
 * @param tree The 1D array representation of the tree.
 * @param n_datapoints Number of data points in `tree`
 * @param n_components Number of components for each data point.
 * @param current_level_start Index of the first element of `tree` which
 *                             contains an element of the current node.
 * @param current_level_nodes Number of elements in this level of the tree (each
 *                             recursive call multiplies it by two).
 * @param start_offset Offset starting from `current_level_start` at which is
 *                      located the root node of the subtree represented by this
 *                      recursive call.
 */
KNode<data_type> *convert_to_knodes(data_type *tree, array_size n_datapoints,
                                    int n_components,
                                    array_size current_level_start,
                                    array_size current_level_nodes,
                                    array_size start_offset);

/**
 * @brief Select an axis to be used for splitting a branch of the tree.
 *
 * @param depth The depth of the tree at this point, might be used in order
 *                to balance the choice of the axes.
 */
inline int select_splitting_dimension(int depth, int n_components) {
  return depth % n_components;
}

/**
 * Sort the given array such that the element in the middle is exactly the
 * median with respect to the given axis, and all the items before and
 * after are respectively lower/greater than that item.

 * @param array Array to be sorted.
 * @param n_datapoints Number of items in array.
 * @param axis Axis along which the sorting must be done.
*/
array_size sort_and_split(DataPoint *array, array_size n_datapoints, int axis);

/**
 * Sort the given vector such that the element in the middle is exactly the
 * median with respect to the given axis, and all the items before and
 * after are respectively lower/greater than that item.
 *
 * @param first_data_point An iterator pointing to the first data point in the
 *                         sequence to be sorted.
 * @param end_data_point An iterator pointing past the last data point in the
 *                         sequence to be sorted.
 * @param axis Axis along which the sorting must be done.
 */
array_size sort_and_split(std::vector<DataPoint>::iterator first_data_point,
                          std::vector<DataPoint>::iterator end_data_point,
                          int axis);

/**
 * @brief Return the given array of values as a vector of `DataPoints`.
 *
 * @param data An array of values of size `n_datapoints*n_components`.
 * @param n_datapoints The number of data points in the dataset.
 * @param n_components The number of components per each data point.
 * @return std::vector<DataPoint>
 */
inline std::vector<DataPoint>
as_data_points(data_type *data, array_size n_datapoints, int n_components) {
  std::vector<DataPoint> data_points;
  if (data == nullptr)
    return data_points;

  data_points.reserve(n_datapoints);
  for (array_size i = 0; i < n_datapoints; i++) {
    data_points.push_back(DataPoint(data + i * n_components));
  }
  return data_points;
}

#ifdef TEST
/**
 * @brief Test that the given k-d tree is built properly.
 *
 * @param root Root of the k-d tree.
 * @param constraints Constraints on this k-d sub-tree given by parent nodes.
 * @param depth Current depth of the k-d tree.
 * @return bool
 */
bool test_kd_tree(KNode<data_type> *root,
                  std::vector<std::optional<data_type>> *constraints,
                  int depth);
#endif
