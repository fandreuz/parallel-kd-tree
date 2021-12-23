#include "data_point.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <vector>

#define TAG_RIGHT_PROCESS_PROCESSING_OVER 10

// holds the rank of whoever called this process
extern int parent;

// number of dimensions in the dataset
extern int dims;

// rank of this process
extern int rank;

// number of MPI processes
extern int rank;

extern int surplus_processes;

// maximum process splitting available for the given number of MPI processes
extern int max_depth;

extern int serial_branch_size;

// list of data point idxes in which this process splitted its branch.
// this process then got assigned the left branch. note that this vector
// contains only "parallel" splits, serial splits are handled otherwise.
extern std::vector<DataPoint> parallel_splits;

// this is an array of pointers since DataPoint resulting from serial splits
// are taken from an already existing DataPoint array
extern DataPoint *serial_splits;

// list of processes started by this process
extern std::vector<int> children;

data_type *generate_kd_tree(data_type *data, int size, int dms, int *new_size);
void build_tree(DataPoint *array, int size, int depth);
void build_tree_serial(DataPoint *array, int size, int depth, int start_index,
                       int right_limit);
// gather results from all children processes and deliver a complete tree
// to the parent process
data_type *finalize(int *new_size);
