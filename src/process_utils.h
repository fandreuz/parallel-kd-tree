#ifndef PUTILS_H
#define PUTILS_H

#include <math.h>

inline int compute_max_depth(int n_processes) {
  return log2((double)n_processes);
}
inline int compute_n_surplus_processes(int n_processes, int max_depth) {
  return n_processes - (int)pow(2.0, (double)max_depth);
}
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

#endif // PUTILS_H
