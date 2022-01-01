#include "tree_printer.h"

std::ostream &print_node_values(std::ostream &os,
                                const KNode<data_type> &node) {
  os << "(";

  for (int i = 0; i < node.get_dims(); i++) {
    if (i > 0)
      os << ",";
    // TODO: cache this somehow in a constexpr
    if (node.get_data(i) == std::numeric_limits<int>::min()) {
      os << "n/a";
      break;
    } else
      os << node.get_data(i);
  }
  return os << ")";
}

std::ostream &print_tree(std::ostream &os, const KNode<data_type> &node,
                         const std::string &prefix, bool isLeft) {
  os << prefix;
  os << (isLeft ? "├──" : "└──");

  // print the value of the node
  print_node_values(os, node);
  os << std::endl;

  // enter the next tree level - left and right branch
  auto left = node.get_left();
  if (left)
    print_tree(os, *left, prefix + (isLeft ? "│   " : "    "), true);
  auto right = node.get_right();
  if (right)
    print_tree(os, *right, prefix + (isLeft ? "│   " : "    "), false);
  return os;
}

std::ostream &operator<<(std::ostream &os, const KNode<data_type> &node) {
  return print_tree(os, node, "", false);
}
