#include "kdtree.h"

data_type *KDTreeGreenhouse::finalize(int *kdtree_size) {
  *kdtree_size = serial_tree_size;
  return unpack_optional_array(serial_tree, serial_tree_size, n_components,
                               EMPTY_PLACEHOLDER);
}
