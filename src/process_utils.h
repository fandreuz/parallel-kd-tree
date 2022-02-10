#pragma once

#include <math.h>

#ifdef USE_MPI
#include <mpi.h>
#elif USE_OMP
#include <omp.h>
#endif

/**
 * @brief Compute the maximum parallel depth reachable with the given set of
 *          processes/threads.
 *
 * Due to the approach used to parallelize the construction of the k-d tree,
 * there is a maximum depth in which every split is guaranteed to have at least
 * one process ready to take on the right branch. This function computes this
 * depth.
 *
 * This function determines indirectly also the number of "surplus" processes.
 *
 * @param n_processes Number of processes/threads available.
 * @return int
 */
inline int compute_max_depth(int n_processes) {
  return log2((double)n_processes);
}

/**
 * @brief Compute the number of surplus processes.
 *
 * A surplus process is a process assigned to a level of the tree which does
 * not guarantee that every splits has a process ready to take on the right
 * branch. Only some splits have a surplus process, starting from the leftmost
 * split in the tree.
 *
 * @param n_processes Number of processes/threads available.
 * @param max_depth Maximum depth which guarantees that there is at least one
 *                    idle process.
 * @return int
 */
inline int compute_n_surplus_processes(int n_processes, int max_depth) {
  return n_processes - (int)pow(2.0, (double)max_depth);
}

/**
 * @brief Compute the rank of the process which is going to take on the right
 *          branch after a split occurred.
 *
 * This function does not assign ranks sequentially. This is an example of the
 * expected output for 10 processes (i.e. max_depth=2).
 * --- level 0 ----
 * 0 -> 4
 * --- level 1 ----
 * 0 -> 2
 * 4 -> 6
 * --- level 2 ----
 * 0 -> 1
 * 2 -> 3
 * 4 -> 5
 * 6 -> 7
 *
 * Then surplus processes come to play:
 * 0 -> 8
 * 1 -> 9
 * 2 -> -1
 * 3 -> -1
 * ...
 *
 * @param rank Rank of the process which operated the split.
 * @param max_depth Maximum depth which guarantees that there is at least one
 *                    idle process.
 * @param next_depth Depth of the next level of the tree (the one after the
 *                    split).
 * @param surplus_processes Number of surplus processes.
 * @param n_processes Number of processes/threads available.
 * @return int
 */
inline int compute_next_process_rank(int rank, int max_depth, int next_depth,
                                     int surplus_processes, int n_processes) {
  // this has two components: one for non-surplus processes, and one for surplus
  if (next_depth < max_depth + 1)
    return rank + pow(2.0, max_depth - next_depth);
  else if (next_depth == max_depth + 1 && rank < surplus_processes)
    return n_processes - surplus_processes + rank;
  else
    return -1;
}

/**
 * @brief Number of parallel workers (i.e. processes for MPI and threads for
 *        OpenMP) that are currently available.
 * @return int
 */
inline int get_n_parallel_workers() {
#ifdef USE_MPI
  int n_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
  return n_processes;
#elif USE_OMP
  return omp_get_num_threads();
#else
  return 1;
#endif
}

/**
 * @brief Return the rank of this process/thread.
 * @return int
 */
inline int get_rank() {
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
#elif USE_OMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}
