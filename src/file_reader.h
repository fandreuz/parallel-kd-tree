#include "tree.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

/*
    Read a CSV file and find the dimensionality of the data points inside.
    This function modifies the given pointer to size and dims according to the
    values determined after parsing the file. It also returns an 1D array
    of data_type which holds the data points. `dims` consecutive values inside
    the returned array represent a data point.
*/
data_type *read_file(std::string filename, int *size, int *dims);
