#pragma once

#include "tree.h"

#include <vector>
#include <cstring>

class DataPoint {
private:
  std::vector<data_type> data;

public:
  DataPoint(data_type *values, int size) { data.assign(values, values + size); }
  data_type operator[](int index) const { return data[index]; }
  void copy_to_array(data_type *array) const {
    std::memcpy(array, data.data(), data.size() * sizeof(data_type));
  }
};

/**
 * @struct
 * @brief Compare numerically two data points along the specified axis.
 */
struct DataPointCompare {
  inline DataPointCompare(size_t index) : index_(index) {}
  inline bool operator()(const DataPoint &dp1, const DataPoint &dp2) const {
    return dp1[index_] < dp2[index_];
  }
  size_t index_;
};
