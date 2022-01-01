#ifndef DATAPOINT_H
#define DATAPOINT_H

#include <cstring>
// this is needed for the value of data_type
#include "tree.h"

// represents a data point
class DataPoint {
  data_type *values = nullptr;
  int data_dimension = -1;

public:
  DataPoint(data_type *dt, int dims);

  // move constructor
  DataPoint(DataPoint &&other);
  // move assignment
  DataPoint &operator=(DataPoint &&other);

  const data_type get(int index) const { return values[index]; }
  data_type *data() { return values; }

  ~DataPoint();
};

struct DataPointCompare {
  inline DataPointCompare(size_t index) : index_(index) {}
  inline bool operator()(const DataPoint &dp1, const DataPoint &dp2) const {
    return dp1.get(index_) < dp2.get(index_);
  }
  size_t index_;
};

#endif // DATAPOINT_H
