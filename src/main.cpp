#include "main_base.h"

template void log_message(std::string);
template void log_message(KNode<data_type> &);

int main(int argc, char **argv) {
  init_parallel_environment(&argc, &argv);

  const std::string filename =
      argc > 1 ? argv[1] : "../benchmark/benchmark1.csv";
  // number of data points and number of components per data point in the
  // dataset
  int n_data_points, n_dims;
  // the dataset as a 1D array, DIMS consecutive items of dt are a data point.
  // with MPI, this reads only on process 0.
  data_type *dt = read_file_serial(filename, &n_data_points, &n_dims);

#ifdef TIME
  double start_time = get_time();
#endif

  // this reference must not be destroyed until we're done reading the tree
  KDTreeGreenhouse greenhouse(dt, n_data_points, n_dims);
  KNode<data_type> *tree = greenhouse.get_grown_kdtree();

#ifdef TIME
  // output the time needed to build the k-d tree
  log_message("# " + std::to_string(get_time() - start_time) + "\n");
#endif

  // we can now delete the data safely
  delete[] dt;

#ifdef OUTPUT
  log_message<KNode<data_type> &>(*tree);
#endif

#ifdef STORE_TO_FILE
  std::string out_filename;
  if (argc > 2)
    write_file(argv[2], tree, n_dims);
  else
    log_message("Path to output file not found.");
#else
  if (argc > 2)
    log_message("You supplied an output file name, but you did not compile "
                "with `make file`.");
#endif

  finalize_parallel_environment();
}
