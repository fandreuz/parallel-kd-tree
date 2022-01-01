#include "tree.h"
#include <iostream>

template class KNode<data_type>;

std::ostream &operator<<(std::ostream &os, const KNode<data_type> &node);
std::ostream &print_tree(std::ostream &os, const KNode<data_type> &node,
                         const std::string &prefix, bool isLeft);
std::ostream &print_node_values(std::ostream &os, const KNode<data_type> &node);
