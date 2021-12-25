# k-d tree
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
