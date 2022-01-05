#include <math.h>

inline int compute_max_depth(int n_processes) {
  return log2((double)n_processes);
}
inline int compute_n_surplus_processes(int n_processes, int max_depth) {
  return n_processes - (int)pow(2.0, (double)max_depth);
}
inline int next_process_rank(int rank, int max_depth, int next_depth) {
  return rank + pow(2.0, max_depth - next_depth);
}
