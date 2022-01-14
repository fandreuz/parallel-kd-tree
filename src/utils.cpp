#include "utils.h"

int bigger_powersum_of_two(int n) {
  int base = 1;
  int N = 0;
  while (N < n) {
    N += base;
    base *= 2;
  }
  return N;
}

int smaller_powersum_of_two(int n) {
  int base = 1;
  int N = 0;
  while (N < n) {
    N += base;
    base *= 2;
  }
  return N - base / 2;
}

// transform the given DataPoint array in a 1D array such that `n_components`
// contiguous items constitute a data point
data_type *unpack_array(DataPoint *array, int size, int n_components) {
  data_type *unpacked = new data_type[size * n_components];
  for (int i = 0; i < size; ++i) {
    array[i].copy_to_array(unpacked + i * n_components, n_components);
  }
  return unpacked;
}

data_type *unpack_array(std::vector<DataPoint>::iterator first_point,
                        std::vector<DataPoint>::iterator last_point,
                        int n_components) {
  data_type *unpacked =
      new data_type[std::distance(first_point, last_point) * n_components];

  int offset = 0;
  for (auto i = first_point; i != last_point; ++i) {
    (*i).copy_to_array(unpacked + offset, n_components);
    offset += n_components;
  }
  return unpacked;
}

// unpack an array which may contain uninitialized items
data_type *unpack_risky_array(DataPoint *array, int size, int n_components,
                              bool *initialized) {
  data_type *unpacked = new data_type[size * n_components];
  for (int i = 0; i < size; ++i) {
    if (initialized[i]) {
      array[i].copy_to_array(unpacked + i * n_components, n_components);
    } else {
      for (int j = 0; j < n_components; ++j) {
        unpacked[i * n_components + j] = EMPTY_PLACEHOLDER;
      }
    }
  }
  return unpacked;
}

data_type *unpack_optional_array(std::optional<DataPoint> *array, int size,
                                 int n_components, data_type fallback_value) {
  data_type *unpacked = new data_type[size * n_components];
  for (int i = 0; i < size; ++i) {
    if (array[i].has_value()) {
      DataPoint dp = std::move(*array[i]);
      dp.copy_to_array(unpacked + i * n_components, n_components);
    } else {
      std::fill_n(unpacked + i * n_components, n_components, fallback_value);
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
void merge_kd_trees(data_type *dest, data_type *branch1, data_type *branch2,
                    int branches_size, int n_components) {
  int already_added = 0;
  // number of nodes in each branch (left and right) at the current level of
  // the tree
  int nodes = 1;
  while (already_added < 2 * branches_size) {
    // we put into the three what's inside the left subtree
    std::memcpy(dest, branch1, nodes * n_components * sizeof(data_type));
    branch1 += nodes * n_components;
    dest += nodes * n_components;

    // we put into the three what's inside the right subtree
    std::memcpy(dest, branch2, nodes * n_components * sizeof(data_type));
    branch2 += nodes * n_components;
    dest += nodes * n_components;

    // we just added left and right branch
    already_added += nodes * 2;
    // the next level will have twice the number of nodes of the current level
    nodes *= 2;
  }
}

#ifdef ALTERNATIVE_SERIAL_WRITE
void rearrange_kd_tree(data_type *dest, data_type *src, int n_datapoints,
                       int n_components) {
  std::vector<int> level_start;
  level_start.push_back(0);
  level_start.push_back(1);

  std::vector<int> nodes_per_level;
  nodes_per_level.push_back(1);
  nodes_per_level.push_back(0);

  // a counter which holds the value of all the powers of 2
  std::vector<int> powers_of_two;
  powers_of_two.push_back(1);
  powers_of_two.push_back(2);

  // we copy the first data point
  std::memcpy(dest, src, n_components * sizeof(data_type));

  int next_node_level = 1;

  for (int i = 1; i < n_datapoints; ++i) {
    int offset =
        level_start[next_node_level] + nodes_per_level[next_node_level];
    std::memcpy(dest + offset * n_components, src + i * n_components,
                n_components * sizeof(data_type));

    nodes_per_level[next_node_level]++;
    if (level_start[next_node_level] + powers_of_two[next_node_level] <
        n_datapoints) {
      ++next_node_level;
    } else {
      for (; nodes_per_level[next_node_level] % 2 == 0 && next_node_level > 0;
           --next_node_level)
        ;
    }

    // we add one whole level to the three vectors
    if ((int)level_start.size() <= next_node_level) {
      nodes_per_level.push_back(0);
      powers_of_two.push_back(powers_of_two[next_node_level - 1] * 2);
      level_start.push_back(level_start[next_node_level - 1] +
                            powers_of_two[next_node_level - 1]);
    }
  }
}
#endif

/*
    Convert the given tree to a linked list structure. This assumes that
    the given size is a powersum of two.

    - tree contains the array representation of the tree
    - size is the number of elements in `tree`
    - n_components is the number of components for each data point
    - current_level_start contains the index of the first element of tree
   which contains an element of the current node
    - current_level_nodes contains the number of elements in this level of the
        tree (each recursive call multiplies it by two)
    - start_offset contains the offset (starting from current_level_start) for
        the root node of the subtree represented by this recursive call.
*/
KNode<data_type> *convert_to_knodes(data_type *tree, int size, int n_components,
                                    int current_level_start,
                                    int current_level_nodes, int start_offset) {
  int next_level_start =
      current_level_start + current_level_nodes * n_components;
  int next_level_nodes = current_level_nodes * 2;
  int next_start_offset = start_offset * 2;

  if (next_level_start < size * n_components) {
    auto left = convert_to_knodes(tree, size, n_components, next_level_start,
                                  next_level_nodes, next_start_offset);
    auto right = convert_to_knodes(tree, size, n_components, next_level_start,
                                   next_level_nodes, next_start_offset + 1);

    return new KNode<data_type>(
        tree + current_level_start + start_offset * n_components, n_components,
        left, right, current_level_start == 0);
  } else
    return new KNode<data_type>(tree + current_level_start +
                                    start_offset * n_components,
                                n_components, nullptr, nullptr, false);
}

int sort_and_split(DataPoint *array, int size, int axis) {
  // the second part of median_idx is needed to unbalance the split towards
  // the left region (which is the one which may parallelize with the highest
  // probability).
  int median_idx = size / 2 - 1 * ((size + 1) % 2);
  std::nth_element(array, array + median_idx, array + size,
                   DataPointCompare(axis));
  // if size is 2 we want to return the first element (the smallest one),
  // since it will be placed into the first empty spot in serial_split
  return median_idx;
}

int sort_and_split(std::vector<DataPoint>::iterator first_data_point,
                   std::vector<DataPoint>::iterator last_data_point, int axis) {
  int size = std::distance(first_data_point, last_data_point);
  // the second part of median_idx is needed to unbalance the split towards
  // the left region (which is the one which may parallelize with the highest
  // probability).
  int median_idx = size / 2 - 1 * ((size + 1) % 2);
  std::nth_element(first_data_point, first_data_point + median_idx,
                   last_data_point, DataPointCompare(axis));
  // if size is 2 we want to return the first element (the smallest one),
  // since it will be placed into the first empty spot in serial_split
  return median_idx;
}
