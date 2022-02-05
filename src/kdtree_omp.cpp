#include "kdtree.h"

data_type *KDTreeGreenhouse::finalize() {
  grown_kdtree_size = tree_size;
  data_type *tree = unpack_optional_array(growing_tree, tree_size, n_components,
                                          EMPTY_PLACEHOLDER);
#ifdef ALTERNATIVE_SERIAL_WRITE
  data_type *temp_tree = new data_type[tree_size * n_components];
  rearrange_kd_tree(temp_tree, tree, tree_size, n_components);
  delete[] tree;
  tree = temp_tree;
#endif
  return tree;
}
