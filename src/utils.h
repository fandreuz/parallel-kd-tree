#include "data_point.h"

#include <cstring>
#include <limits>

#define EMPTY_PLACEHOLDER std::numeric_limits<int>::min()

/* return the nearest number N such that N > n and N is a sum of powers of two
   Example:
    5 -> 7 = 1 + 2 + 4
    3 -> 3 = 1 + 2
*/
inline int bigger_powersum_of_two(int n) {
  int base = 1;
  int N = 0;
  while (N < n) {
    N += base;
    base *= 2;
  }
  return N;
}

inline int smaller_powersum_of_two(int n) {
  int base = 1;
  int N = 0;
  while (N < n) {
    N += base;
    base *= 2;
  }
  return N - base / 2;
}

// transform the given DataPoint array in a 1D array such that `dims` contiguous
// items constitute a data point
inline data_type *unpack_array(DataPoint *array, int size, int dims) {
  data_type *unpacked = new data_type[size * dims];
  for (int i = 0; i < size; ++i) {
    data_type *d = array[i].data();
    std::memcpy(unpacked + i * dims, d, dims * sizeof(data_type));
  }
  return unpacked;
}

// unpack an array which may contain uninitialized items
inline data_type *unpack_risky_array(DataPoint *array, int size, int dims,
                                     bool *initialized) {
  data_type *unpacked = new data_type[size * dims];
  for (int i = 0; i < size; ++i) {
    if (initialized[i]) {
      data_type *d = array[i].data();
      std::memcpy(unpacked + i * dims, d, dims * sizeof(data_type));
    } else {
      for (int j = 0; j < dims; ++j) {
        unpacked[i * dims + j] = EMPTY_PLACEHOLDER;
      }
    }
  }
  return unpacked;
}

/*
  This function rearranges branch1 and branch2 into dest such that we first
  take 1 node from branch1 and 1 node from branch2, then 2 nodes from branch1
  and 2 nodes from branch2, then 4 nodes from branch1 and 4 nodes from
  branch2..

  Note that this function is dimensions-safe (i.e. copies all the dimensions).

  Remember to add a split point before this function call (if you need to).
*/
inline void rearrange_branches(data_type *dest, data_type *branch1,
                               int branch1_size, data_type *branch2,
                               int branch2_size, int dims) {
  int already_added = 0;
  // number of nodes in each branch (left and right)at the current level of
  // the tree
  int nodes = 1;
  // index of left(right)_branch_buffer from which we start memcpying
  int start_index = 0;
  while (already_added < branch1_size + branch2_size) {
    // we put into the three what's inside the left subtree
    if (branch1_size > 0) {
      std::memcpy(dest + already_added * dims, branch1 + start_index,
                  nodes * dims * sizeof(data_type));
    }
    // we put into the three what's inside the right subtree
    std::memcpy(dest + (nodes + already_added) * dims, branch2 + start_index,
                nodes * dims * sizeof(data_type));

    // the next iteration we're going to start in a different position of
    // left(right)_branch_buffer
    start_index += nodes * dims;
    // we just added left and right branch
    already_added += nodes * 2;
    // the next level will have twice the number of nodes of the current level
    nodes *= 2;
  }
}
