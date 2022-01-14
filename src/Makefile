# Makefile for the K-D tree.
#
# make debug	 ----> Show both the output and debug messages during the
# 				       construction of the tree.
# make output    ----> Show the binary tree (and nothing else).
# make time      ----> Produces an executable which prints only the time taken
#				 	   to build the tree.
# make leaks	 ----> Produces an executable which can be used to find find
#					   memory leaks.
# make mpidebug  ----> Enables debugging via gdb (the rank of the debugged
#					   process must be set via the environment variable
#					   MPI_DEBUG_RANK).
# make file      ----> Save the k-d tree as a CSV file in the current directory,
#					   in order of increasing level (left to right).
# make clean	 ----> Clean the folder (removes executable/precompiled headers
#		   	           files)

.DEFAULT_GOAL := output
FLAGS = -std=c++17 -Wall -Wextra
UTILITIES = main.cpp file_utils.cpp tree_printer.cpp utils.cpp kdtree.cpp

# OpenMP or MPI?
ifeq ($(src), mpi)
	SRC = kdtree_mpi.cpp
	ADDITIONAL_FLAGS = -D USE_MPI
	COMPILER = mpic++
	OUT = tree_mpi.x
else
	SRC = kdtree_omp.cpp
	ADDITIONAL_FLAGS = -fopenmp
	COMPILER = g++
	OUT = tree_omp.x
endif

# alternative writing mode?
ifeq ($(write), alternative)
	WRITE_MODE_FLAG = -D ALTERNATIVE_SERIAL_WRITE
else
	WRITE_MODE_FLAG =
endif

compile:
	$(COMPILER) $(FLAGS) $(WRITE_MODE_FLAG) $(ADDITIONAL_FLAGS) -O3 $(UTILITIES) $(SRC) -o $(OUT)

debug:
	$(COMPILER) $(FLAGS) $(WRITE_MODE_FLAG) $(ADDITIONAL_FLAGS) -D DEBUG -D OUTPUT $(UTILITIES) $(SRC) -o $(OUT)

output:
	$(COMPILER) $(FLAGS) $(WRITE_MODE_FLAG) $(ADDITIONAL_FLAGS) -D OUTPUT $(UTILITIES) $(SRC) -o $(OUT)

time:
	$(COMPILER) $(FLAGS) $(WRITE_MODE_FLAG) $(ADDITIONAL_FLAGS) -D TIME -O3 $(UTILITIES) $(SRC) -o $(OUT)

leaks:
	$(COMPILER) $(FLAGS) $(WRITE_MODE_FLAG) $(ADDITIONAL_FLAGS) -static-libasan -fsanitize=address -g -D DEBUG $(UTILITIES) $(SRC) -o $(OUT)

mpidebug:
	$(COMPILER) $(FLAGS) $(WRITE_MODE_FLAG) $(ADDITIONAL_FLAGS) -g -D DEBUG -D MPI_DEBUG $(UTILITIES) $(SRC) -o $(OUT)

file:
	$(COMPILER) $(FLAGS) $(WRITE_MODE_FLAG) $(ADDITIONAL_FLAGS) -D STORE_TO_FILE $(UTILITIES) $(SRC) -o $(OUT)

clean:
	rm *.x
	rm *.gch
