#pragma once

#include "knode.h"

#include <iostream>

template class KNode<data_type>;

/**
 * @brief Auxiliary function which pretty-prints a k-d tree.
 *
 * This auxiliary function prints the given k-d tree in a pretty way, i.e.
 * the tree develops horizontally on the screen to exploit the space available
 * in the best possible way.
 *
 * @param os   The output file descriptor.
 * @param node The root of the k-d tree to be printed.
 * @return std::ostream& The output file descriptor.
 */
std::ostream &operator<<(std::ostream &os, const KNode<data_type> &node);

/**
 * @brief Print the content of the given node.
 *
 * @param os   The output file descriptor.
 * @param node The root of the k-d tree to be printed.
 * @return std::ostream& The output file descriptor.
 */
std::ostream &print_node_values(std::ostream &os, const KNode<data_type> &node);
