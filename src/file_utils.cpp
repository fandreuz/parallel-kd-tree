#include "file_utils.h"

/*
    This function reads the file row by row, and for each row stores the
    numbers found in an std::vector. The accepted dimension of each data point
    is the dimension of the data point in the first row of the file (a check
    is performed for each row though).

    When there are no more rows, we copy the content of the vector on a 1D
    array (via std::memcpy) and return that 1D array to the user.
 */
data_type *read_file(const std::string &filename, int *size, int *dims) {
  std::ifstream file(filename);

  // local variable which holds the last known number of dimensions per data
  // point. used to check that all data points have the same number of
  // components
  int temp_dims = -1;

  constexpr char separator = ',';

  std::vector<data_type> lines_buffer;
  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      std::vector<data_type> row_buffer;

      int start = 0;
      for (int i = 1; i < line.length(); ++i) {
        // we found a component
        if (line[i] == separator) {
          row_buffer.push_back(string_converter(line.substr(start, i)));
          start = i + 1;
        }
      }
      row_buffer.push_back(string_converter(line.substr(start, line.length())));

      // we check that all the components have the same number of dimensions
      if (temp_dims != -1 && temp_dims != row_buffer.size())
        throw std::invalid_argument(
            "Invalid number of dimensions for data point number " +
            std::to_string(lines_buffer.size()));
      temp_dims = row_buffer.size();

      // we put evertthing into line_buffer
      for (int i = 0; i < temp_dims; ++i)
        lines_buffer.push_back(row_buffer[i]);
    }
    file.close();
  } else {
    throw std::invalid_argument("File not found.");
  }

  int temp_size = lines_buffer.size() / temp_dims;

  // we return the number of data points and their dimensionality
  *dims = temp_dims;
  *size = temp_size;

#ifdef DEBUG
  std::cout << "Found " << *size << " data points with " << *dims
            << " dimensions" << std::endl;
#endif

  data_type *data_points = new data_type[temp_size * temp_dims];
  std::memcpy(data_points, lines_buffer.data(),
              temp_size * temp_dims * sizeof(data_type));

  return data_points;
}
