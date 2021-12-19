#include "tree_mpi.cpp"
#include <iostream>

int main() {
  data_type dt[100];
  for (int i = 0; i < 100; i++) {
    dt[i] = i * i - 2 * i;
  }

  int *tree = generate_2d_tree(dt, 100);
  KNode *knode_tree = as_knode(tree);
  delete[] tree;

  for (int i = 0; i < 100; i++) {
    std::cout << tree[i] << std::endl;
  }
}
