# Makefile for the K-D tree.
#
# make compile	 ----> Compile the source code.
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
# make clean	 ----> Clean the folder (removes executable/precompiled headers
#		   	           files)

.DEFAULT_GOAL := output
FLAGS = -std=c++14 -Wall -Wextra
UTILITIES = data_point.cpp main.cpp file_reader.cpp tree_printer.cpp utils.cpp

# OpenMP or MPI?
ifeq ($(src), mpi)
	SRC = kdtree_mpi.cpp
	OPENMP_FLAGS =
	COMPILER = mpic++
else
	SRC = kdtree_openmp.cpp
	OPENMP_FLAGS = -fopenmp
	COMPILER = g++
endif

compile:
	mpic++ $(FLAGS) $(OPENMP_FLAGS) -O3 $(UTILITIES) $(SRC) -o tree_mpi.x

debug:
	mpic++ $(FLAGS) $(OPENMP_FLAGS) -D DEBUG -D OUTPUT $(UTILITIES) $(SRC) -o tree_mpi.x

output:
	mpic++ $(FLAGS) $(OPENMP_FLAGS) -D OUTPUT $(UTILITIES) $(SRC) -o tree_mpi.x

time:
	mpic++ $(FLAGS) $(OPENMP_FLAGS) -D TIME -O3 $(UTILITIES) $(SRC) -o tree_mpi.x

leaks:
	mpic++ $(FLAGS) $(OPENMP_FLAGS) -static-libasan -fsanitize=address -g -D DEBUG $(UTILITIES) $(SRC) -o tree_mpi.x

mpidebug:
	mpic++ $(FLAGS) $(OPENMP_FLAGS) -g -D DEBUG -D MPI_DEBUG $(UTILITIES) $(SRC) -o tree_mpi.x

clean:
	rm *.x
	rm *.gch
