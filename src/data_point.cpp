#include "data_point.h"

DataPoint::DataPoint(data_type *dt, int dims)
    : values{new data_type[dims]}, data_dimension{dims} {
  // we need to copy the values since a call to std::nth_element changes the
  // order of the array, therefore pointers do not point anymore to the
  // values we expected
  std::memcpy(values, dt, data_dimension * sizeof(data_type));
}

DataPoint::DataPoint(DataPoint &&other) {
  if (this != &other) {
    values = other.values;

    data_dimension = other.data_dimension;

    other.values = nullptr;
    other.data_dimension = -1;
  }
}

DataPoint &DataPoint::operator=(DataPoint &&other) {
  if (this != &other) {
    data_type *vls = other.values;
    delete[] values;
    values = vls;

    data_dimension = other.data_dimension;

    other.values = nullptr;
    other.data_dimension = -1;
  }

  return *this;
}

DataPoint::~DataPoint() {
  delete[] values;
  values = nullptr;
  data_dimension = -1;
}
