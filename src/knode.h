#pragma once

#include <iostream>
#include <limits>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifndef USE_DOUBLE_PRECISION_DATA
/**
 * @def
 * @brief The datatype of the k-d tree.
 */
#define data_type float
/**
 * @def
 * @brief A function used to convert from string to `data_type`.
 */
#define string_converter std::stof

#ifdef USE_MPI
/**
 * @def
 * @brief MPI version of `data_type`.
 */
#define mpi_data_type MPI_FLOAT
#endif

#else
/**
 * @def
 * @brief The datatype of the k-d tree.
 */
#define data_type double
/**
 * @def
 * @brief A function used to convert from string to `data_type`.
 */
#define string_converter std::stod

#ifdef USE_MPI
/**
 * @def
 * @brief MPI version of `data_type`.
 */
#define mpi_data_type MPI_DOUBLE
#endif

#endif

#ifdef BIG
using array_size = std::size_t;
#else
using array_size = int;
#endif

/**
 * @class
 * @brief A node of a k-d tree.
 *
 * A node represents a data point used to split the dataset at a certain point.
 * Each node holds the associated data point.
 *
 * **Important**: When constructin a k-d tree, we assume that the data points
 * are stored in a 1D array ready to be transformed into a kind-of linked list.
 * For this reason, to avoid a lot of memory allocations, we use that 1D
 * array and pointers to the internal cells of that array as pointers to the
 * first component of a data point.
 *
 * For this reason:
 *
 *  1. Users must not delete data given to the constructor of a `KNode`;
 *  2. Users must not modify data given to the constructor of a `KNode`.
 *
 * To delete a k-d tree you just need to delete its root node. Make sure that
 * you do not delete data passed to the constructor of a `KNode` in any part of
 * your code, since the root node will take care to delete it for you.
 *
 * @tparam T The type of the values which compose the points in the
 *          dataset.
 */
template <typename T> class KNode {
  T *data;  /**< Data point associated with this node, as 1D array. */
  int dims; /**< Number of dimensions of the data point (i.e. size of `data`).
             */

  KNode<T> *left, /**< Root of the left branch originating from this node. */
      *right;     /**< Root of the right branch originating from this node. */

  bool is_root; /**< If `true`, this node is considered to be the root node, and
                      therefore is responsible to delete the dataset. */

public:
  /**
   * @brief Construct a new `KNode`.
   *
   * @param d A pointer to the first item of the data point associated with this
   *           node.
   * @param dms The number of components of a data point.
   * @param l A pointer to the root of the left branch originating from this
   *           node.
   * @param r A pointer to the root of the right branch originating from this
   *           node.
   * @param root The user should pass `true` if this node is the root node,
   *           `false` otherwise.
   */
  KNode(data_type *d, int dms, KNode<T> *l, KNode<T> *r, bool root)
      : data{d}, dims{dms}, left{l}, right{r}, is_root{root} {}

  KNode(KNode<T> &&other)
      : data{other.data}, dims{other.dims}, left{other.left},
        right{other.right}, is_root{other.is_root} {
    other.data = nullptr;
    other.left = nullptr;
    other.right = nullptr;
    other.dims = -1;
  }

  KNode()
      : data{nullptr}, dims{0}, left{nullptr}, right{nullptr}, is_root{false} {}

  /**
   * @brief Destroy the KNode object, and also the dataset if `is_root` is
   *         `true`.
   */
  ~KNode() {
    if (is_root)
      delete[] data;
    delete left;
    delete right;
  }

  /**
   * @brief Get the value of the data point on the i-th axis.
   */
  T get_data(int i) const { return data[i]; }
  /**
   * @brief Get the number of axes in the dataset.
   */
  int get_dims() const { return dims; }

  /**
   * @brief Get a pointer to the root of the left branch originating from this
   *         node.
   */
  KNode<T> *get_left() const { return left; }
  /**
   * @brief Get a pointer to the root of the right branch originating from this
   *         node.
   */
  KNode<T> *get_right() const { return right; }
};
