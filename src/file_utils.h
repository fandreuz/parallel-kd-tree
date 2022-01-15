#pragma once

#include "knode.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <queue>
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
data_type *read_file(const std::string &filename, std::size_t *size, int *dims);

/**
 * @brief Write a k-d tree to a CSV file.
 *
 * The k-d tree will be stored as a CSV file where each row represents a data
 * point. The k-d tree is stored in order of increasing leve, left to right.
 *
 * It is recommended that this library and the user which is going to use the
 * file agree on the splitting axes used to construct the k-d tree.
 *
 * @param filename Path (relative or absolute) to the CSV file.
 * @param KNode*   Root of the tree to be stored.
 * @param dims     Number of components per each data point.
 */
template <typename T>
void write_file(const std::string &filename, KNode<T> *root, int dims) {
  std::ofstream outdata;
  outdata.open(filename, std::fstream::out);
  if (!outdata) {
    throw std::invalid_argument("File not found.");
  }

  std::queue<KNode<T> *> to_visit;
  to_visit.push(root);

  while (to_visit.size() > 0) {
    KNode<T> *node = to_visit.front();
    to_visit.pop();

    for (int i = 0; i < dims; ++i) {
      outdata << node->get_data(i);
      if (i < dims - 1)
        outdata << ",";
    }

    if (node->get_left() != nullptr)
      to_visit.push(node->get_left());
    if (node->get_right() != nullptr)
      to_visit.push(node->get_right());

    outdata << std::endl;
  }
}
