#pragma once

#include "knode.h"

#include <cstring>

class DataPoint {
private:
#ifdef IMPROVED_MEMORY_ACCESS
  data_type **scattered_data = nullptr;
#endif
  data_type *data = nullptr;

public:
  DataPoint(data_type *values) : data{values} {}
  data_type get(int index) const noexcept { return data[index]; }

#ifdef IMPROVED_MEMORY_ACCESS
  DataPoint(data_type *values, data_type **scattered_values) {
    data = values;
    scattered_data = scattered_values;
  }
  data_type get_scattered(int index) const noexcept {
    return *scattered_data[index];
  }
#endif

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
#ifdef IMPROVED_MEMORY_ACCESS
    return dp1.get_scattered(index_) < dp2.get_scattered(index_);
#else
    return dp1.get(index_) < dp2.get(index_);
#endif
  }
  array_size index_;
};
