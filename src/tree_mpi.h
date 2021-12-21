#include <algorithm>
#include <cstring>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <vector>

#if !defined(DOUBLE_PRECISION)
#define mpi_data_type MPI_FLOAT
#define data_type float
#else
#define mpi_data_type MPI_DOUBLE
#define data_type double
#endif

#define TAG_RIGHT_PROCESS_PROCESSING_OVER 10

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
      delete[] values;

      values = other.values;
      data_dimension = other.data_dimension;
    }
  }

  inline DataPoint& operator=(const DataPoint& other) {
    values = other.values;
    data_dimension = other.data_dimension;
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
};

struct DataPointCompare {
  inline DataPointCompare(size_t index) : index_(index) {}
  inline bool operator()(const DataPoint &dp1, const DataPoint &dp2) const {
    return dp1.get(index_) < dp2.get(index_);
  }
  size_t index_;
};

// holds the rank of whoever called this process
extern int parent;

// number of dimensions in the dataset
extern int dims;

// rank of this process
extern int rank;

// maximum process splitting available for the given number of MPI processes
extern int max_depth;

// list of data point idxes in which this process splitted its branch.
// this process then got assigned the left branch. note that this vector
// contains only "parallel" splits, serial splits are handled otherwise.
extern std::vector<int> parallel_splits;

// this is an array of pointers since DataPoint resulting from serial splits
// are taken from an already existing DataPoint array
extern DataPoint *serial_splits;

// list of processes started by this process
extern std::vector<int> children;
// for each child, the size of the branch assigned to that child. used in
// finalize to know what to expect from my children
extern std::vector<int> right_branch_sizes;
extern std::vector<int> left_branch_sizes;
