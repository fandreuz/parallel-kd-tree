#include <cstring>
#include <iostream>

#if !defined(DOUBLE_PRECISION)
#define mpi_data_type MPI_FLOAT
#define data_type float
#else
#define mpi_data_type MPI_DOUBLE
#define data_type double
#endif

// represents a data point
class DataPoint {
  data_type *values = nullptr;
  int data_dimension = -1;

public:
  inline DataPoint(data_type *dt, int dims) {
    // we need to copy the values since a call to std::nth_element changes the
    // order of the array, therefore pointers do not point anymore to the
    // values we expected
    values = new data_type[dims];
    std::memcpy(values, dt, dims * sizeof(data_type));

    data_dimension = dims;
  }

  inline DataPoint(DataPoint &&other) {
    if (this != &other) {
      data_type *vls = other.values;
      delete[] values;
      values = vls;

      data_dimension = other.data_dimension;

      other.values = nullptr;
      other.data_dimension = -1;
    }
  }

  // copy assignment
  inline DataPoint &operator=(const DataPoint &other) {
    data_dimension = other.data_dimension;

    values = new data_type[data_dimension];
    std::memcpy(values, other.values, data_dimension * sizeof(data_type));

    return *this;
  }

  inline const data_type get(int index) const {
#ifdef NONSAFE
    if (index < data_dimension)
      return values[index];
    else
      return -1;
#else
    return values[index];
#endif
  }

  data_type *data() { return values; }

  ~DataPoint() {
    std::cerr << "Deleting data point, value address: " << values << std::endl;
    delete[] values;
    data_dimension = -1;
  }
};

struct DataPointCompare {
  inline DataPointCompare(size_t index) : index_(index) {}
  inline bool operator()(const DataPoint &dp1, const DataPoint &dp2) const {
    return dp1.get(index_) < dp2.get(index_);
  }
  size_t index_;
};
