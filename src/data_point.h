#ifndef DATAPOINT_H
#define DATAPOINT_H

#include <cstring>
// this is needed for the value of data_type
#include "tree.h"

/**
 * @class
 * @brief Represents a data point.
 */
class DataPoint {
  data_type *values =
      nullptr; /**< Pointer to the first element of this data point. */
  int data_dimension = -1; /**< Number of items in this data point. */

public:
  /**
   * @brief Construct a new Data Point object.
   *
   * @param dt Pointer to the first element of this data point.
   * @param dims Number of items in this data point.
   */
  DataPoint(data_type *dt, int dims);

  /**
   * @brief Move constructor.
   */
  DataPoint(DataPoint &&other);
  /**
   * @brief Move assignment.
   */
  DataPoint &operator=(DataPoint &&other);

  /**
   * @brief Get the value of the data point on the i-th axis.
   */
  data_type get(int index) const { return values[index]; }
  /**
   * @brief Get a pointer to the first element of this data point.
   */
  data_type *data() { return values; }

  ~DataPoint();
};

/**
 * @struct
 * @brief Compare numerically two data points along the specified axis.
 */
struct DataPointCompare {
  inline DataPointCompare(size_t index) : index_(index) {}
  inline bool operator()(const DataPoint &dp1, const DataPoint &dp2) const {
    return dp1.get(index_) < dp2.get(index_);
  }
  size_t index_;
};

#endif // DATAPOINT_H
