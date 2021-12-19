#ifdef MPI_VERSION
#include <mpi.h>
#endif

#define NDIM 2

#if !defined(DOUBLE_PRECISION)
#define data_type float

#ifdef MPI_VERSION
#define mpi_data_type MPI_FLOAT
#endif

#else
#define data_type double

#ifdef MPI_VERSION
#define mpi_data_type MPI_FLOAT
#endif

#endif

// representation of an n-dimensional data point
typedef kpoint data_type[NDIM];

struct KNode {
    // point used to split the tree
    kpoint data;
    // left and right branch originating from this node
    KNode *left = nullptr, *right = nullptr;
};
