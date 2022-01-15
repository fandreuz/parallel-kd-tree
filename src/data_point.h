#pragma once

#include "knode.h"

#include <cstring>

class DataPoint {
private:
  data_type *data;

public:
  DataPoint(data_type *values) { data = values; }
  data_type operator[](int index) const { return data[index]; }
  void copy_to_array(data_type *array, int n_components) const {
    std::memcpy(array, data, n_components * sizeof(data_type));
  }
};

/**
 * @struct
 * @brief Compare numerically two data points along the specified axis.
 */
struct DataPointCompare {
  inline DataPointCompare(array_size index) : index_(index) {}
  inline bool operator()(const DataPoint &dp1, const DataPoint &dp2) const {
    return dp1[index_] < dp2[index_];
  }
  array_size index_;
};
