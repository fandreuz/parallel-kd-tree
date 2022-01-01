#include "data_point.h"
#include "tree.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
#include <vector>

/*
  Generate a k-d tree from the given data. If this process is not the main
  process, this function blocks the process until another process wakes
  this process with something to process.

  - data is a 1D array of data such that k consecutive items constitute a data
        point;
  - size is the dimension of the dataset, i.e. len(data) / dms;
  - dms is the number of components for each data point.
*/
KNode<data_type> *generate_kd_tree(data_type *data, int size, int dms);
