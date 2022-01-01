#ifndef UTILS_H
#define UTILS_H

#include "data_point.h"
#include "tree.h"

#include <cstring>
#include <limits>

#define EMPTY_PLACEHOLDER std::numeric_limits<int>::min()

/* return the nearest number N such that N >= n and N is a sum of powers of two
   Example:
    5 -> 7 = 1 + 2 + 4
    3 -> 3 = 1 + 2
*/
int bigger_powersum_of_two(int n);

/* return the nearest number N such that N <= n and N is a sum of powers of two
   Example:
    5 -> 7 = 1 + 2 + 4
    3 -> 3 = 1 + 2
*/
int smaller_powersum_of_two(int n);

// transform the given DataPoint array in a 1D array such that `dims` contiguous
// items constitute a data point
data_type *unpack_array(DataPoint *array, int size, int dims);

// unpack an array which may contain uninitialized items
data_type *unpack_risky_array(DataPoint *array, int size, int dims,
                              bool *initialized);

/*
  This function rearranges branch1 and branch2 into dest such that we first
  take 1 node from branch1 and 1 node from branch2, then 2 nodes from branch1
  and 2 nodes from branch2, then 4 nodes from branch1 and 4 nodes from
  branch2..

  Note that this function is dimensions-safe (i.e. copies all the dimensions).

  Remember to add a split point before this function call (if you need to).
*/
void rearrange_branches(data_type *dest, data_type *branch1, data_type *branch2,
                        int branches_size, int dims);

/*
    Convert the given tree to a linked list structure. This assumes that
    the given size is a powersum of two.

    - tree contains the array representation of the tree
    - size is the number of elements in `tree`
    - dims is the number of components for each data point
    - current_level_start contains the index of the first element of tree which
        contains an element of the current node
    - current_level_nodes contains the number of elements in this level of the
        tree (each recursive call multiplies it by two)
    - start_offset contains the offset (starting from current_level_start) for
        the root node of the subtree represented by this recursive call.
*/
KNode<data_type> *convert_to_knodes(data_type *tree, int size, int dims,
                                    int current_level_start,
                                    int current_level_nodes, int start_offset);

#endif // UTILS_H
