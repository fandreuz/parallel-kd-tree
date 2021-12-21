#include "tree_mpi.h"
#include <iostream>

int main() {
  data_type dt[100];
  for (int i = 0; i < 100; i++) {
    dt[i] = i * i - 2 * i;
  }

  int *tree = generate_kd_tree(dt, 100, 2);
}
