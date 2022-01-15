#include "file_utils.h"

/*
    This function reads the file row by row, and for each row stores the
    numbers found in an std::vector. The accepted dimension of each data point
    is the dimension of the data point in the first row of the file (a check
    is performed for each row though).

    When there are no more rows, we copy the content of the vector on a 1D
    array (via std::memcpy) and return that 1D array to the user.
 */
data_type *read_file(const std::string &filename, std::size_t *size,
                     int *dims) {
  std::ifstream file(filename);

  // local variable which holds the last known number of dimensions per data
  // point. used to check that all data points have the same number of
  // components
  std::optional<std::size_t> last_known_dims = std::nullopt;

  constexpr char separator = ',';

  std::vector<data_type> lines_buffer;
  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      std::vector<data_type> row_buffer;

      int start = 0;
      for (std::size_t i = 1; i < line.length(); ++i) {
        // we found a component
        if (line[i] == separator) {
          row_buffer.push_back(string_converter(line.substr(start, i)));
          start = i + 1;
        }
      }
      row_buffer.push_back(string_converter(line.substr(start, line.length())));

      // we check that all the components have the same number of dimensions
      if (last_known_dims.has_value() && *last_known_dims != row_buffer.size())
        throw std::invalid_argument(
            "Invalid number of dimensions for data point number " +
            std::to_string(lines_buffer.size()));
      last_known_dims.emplace(row_buffer.size());

      // we put evertthing into line_buffer
      for (std::size_t i = 0; i < *last_known_dims; ++i)
        lines_buffer.push_back(row_buffer[i]);
    }
    file.close();
  } else {
    throw std::invalid_argument("File not found.");
  }

  // we return the number of data points and their dimensionality
  std::size_t loc_size = lines_buffer.size() / *last_known_dims;
  *size = loc_size;
  int loc_dims = (int)*last_known_dims;
  *dims = loc_dims;

#ifdef DEBUG
  std::cout << "Found " << *size << " data points with " << *dims
            << " dimensions" << std::endl;
#endif

  data_type *data_points = new data_type[loc_size * loc_dims];
  std::memcpy(data_points, lines_buffer.data(),
              loc_size * loc_dims * sizeof(data_type));

  return data_points;
}
