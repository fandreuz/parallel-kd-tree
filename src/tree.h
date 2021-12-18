#define NDIM 2

#if !defined(DOUBLE_PRECISION)
#define data_type float
#else
#define data_type double
#endif

// representation of an n-dimensional data point
typedef kpoint data_type[NDIM];

struct KNode {
    // index of the dimension used to split the current branch in this node
    int axis;
    // point used to split the tree
    kpoint data;
    // left and right branch originating from this node
    KNode *left = nullptr, *right = nullptr;
}
