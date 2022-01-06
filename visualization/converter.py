from visualize_kd_tree import KDTreeNode
from csv import reader

# build a KDTree recursively by exploring the given array
def build_knode(arr, level_start, level_width, offset):
    if level_start + offset >= len(arr):
        return None
    if arr[level_start + offset][0] < -2 * 1e9:
        return None

    return KDTreeNode(
        arr[level_start + offset],
        build_knode(
            arr, level_start + level_width, level_width * 2, offset * 2
        ),
        build_knode(
            arr, level_start + level_width, level_width * 2, offset * 2 + 1
        ),
    )


# convert a CSV file to a KDTree
def csv_to_tree(filename):
    items = []
    with open(filename, newline="") as csvfile:
        rd = reader(csvfile, delimiter=",")
        items.extend(map(tuple, [map(float, l) for l in rd]))
    return build_knode(items, 0, 1, 0)
