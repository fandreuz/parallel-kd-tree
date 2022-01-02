#include "tree.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Extract a dataset of k-dimensional data points from a CSV file.
 *
 * Extract a dataset of k-dimensional data points from the rows of a given
 * CSV file. Each row should contain exactly one data point, the number of
 * components for each data point is assumed to be constant. The separator
 * between the components is assumed to be a single comma (',').
 *
 * This function also sets the variables `size` and `dims` to retrieve
 * respectively the number of data points in the dataset and the number of
 * components for each data point.
 *
 * @param filename Path (relative or absolute) to the CSV file.
 * @param size     A pointer to a variable which is going to contain the number
 *                  of data points in the dataset when this function returns
 *                  succesfully.
 * @param dims     A pointer to a variable which is going to contain the number
 *                  of components for each data point when this function returns
 *                  succesfully.
 * @return data_type* A 1D array whose size is `size*dims`, where `dims`
 *                      consecutive items represent a data point.
 */
data_type *read_file(std::string filename, int *size, int *dims);
