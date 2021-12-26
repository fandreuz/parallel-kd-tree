# Parallel k-d tree
Implementation of a **parallel** *k-d tree* using MPI and OpenMP.

## Summary
A *k-d tree* is a data structure which can be used to represent k-dimensional
points in a convenient way. This kind of representation can be used to implement
algorithms like K-Nearest Neighbors (KNN) very efficiently [2].

The procedure employed to build such a structure is the following:

1. Choose one axis;
2. Determine the median point for the dataset with respect to that axis
   (i.e. the one which splits the dataset in two halves along the chosen axis);
3. For each one of the two halves, go back to `1`.

The algorithms stops when the dataset contains one point.

A k-d tree may be represented with a binary tree, whose nodes
should be defined approximately in the following way:

```cpp
struct Node {
    // k-dimensional value
    double *data;

    Node *left;
    Node *right;
};
```
We assume that our data can be represented using a `double` value.

## Parallelization
### MPI
The strategy employed for the parallelization with MPI consists in delegating
one of the two branches resulting from a split to one of the available processes
waiting. At the beginning we start with only one process (which we refer to as
'`rank0`') processing the data. After the split (step `2`) `rank0` produces two
branches. The right branch (data greater than the median along the chosen axis)
is assigned to a new process ('`rank1`'), which then carries out the computation
independently. The left branch is kept by `rank0`.

This same strategy is then employed recursively by `rank0` and `rank1`, until
the tree is completed.

## Example
Given the following input:
```
(0,-2,-4) / (1,-1,-3) / (2,0,-2) / (3,1,-1) / (4,2,0)
```

The k-d tree produced is:
```
└──(2,0,-2)
    ├──(0,-2,-4)
    │   ├──(n/a)
    │   └──(1,-1,-3)
    └──(3,1,-1)
        ├──(n/a)
        └──(4,2,0)
```

## Compile
Download the source code with `git clone https://github.com/fAndreuzzi/kd-tree.git`,
then navigate to the `src` folder and compile the source using the makefile.
The following recipes are available:
- `make compile`: Compile the source code, the binary produced won't produce any
  kind of output;
- `make debug`: The binary produced will show debug messages and the output;
- `make output`: Show only the output;
- `make time`: Show only the time taken to build the k-d tree;
- `make leaks`: Find memory leaks in the source code, does not produce any other
  output;
- `make mpidebug`: Prepare a binary that can be debugged using gdb (the rank
  of the process to be controlled via gdb must be set via the environment
  variable `MPI_DEBUG_RANK`).

## Usage
You can run the executable `tree_mpi.x` generated in the step
[Compile](#compile) using the usual command `mpirun`:

### Serial
`mpirun -np 1 tree_mpi.x`

### 10 processors
`mpirun -np 10 tree_mpi.x`

### Specify a dataset
By default the dataset used is the file `benchmark/benchmar1.csv`. You can
specify your own dataset via a command line argument. Valid datasets are CSV
files where each data point has the same number of components (one data point
per row).

For example, the following command runs the k-d tree algorithm on the dataset
inside the file `foo.csv` in the current directory:

`mpirun -np 10 tree_mpi.x foo.csv`


## Roadmap
- [x] Working MPI implementation;
  - [ ] Optimize the last call to `finalize()`: maybe it's not needed (since we traverse the tree in `utils.convert_to_knodes()`);
  - [ ] Fix some memory leaks.
- [ ] Working MP implementation;
- [ ] Visual representation of the tree;
- [ ] Performance evaluation.
- [ ] Comparison against some serial implementation(?)

## References
1. Friedman, Jerome H., Jon Louis Bentley, and Raphael Ari Finkel. ["An algorithm for finding best matches in logarithmic expected time."](https://homes.di.unimi.it/righini/Didattica/AlgoritmiEuristici/MaterialeAE/Friedman%20k-d%20trees.pdf) ACM Transactions on Mathematical Software (TOMS) 3.3 (1977): 209-226.
2. [Wikipedia](https://en.wikipedia.org/wiki/K-d_tree)
