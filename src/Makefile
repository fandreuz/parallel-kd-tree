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
MAIN = main.x
FLAGS = -std=c++14 -Wall -Wextra
SOURCE = data_point.cpp tree_mpi.cpp main.cpp file_reader.cpp tree_printer.cpp tree.h utils.cpp
# eliminate default suffixes
.SUFFIXES:
SUFFIXES=

.SUFFIXES: .cpp .x .h

compile: main.cpp tree_mpi.cpp data_point.cpp
	mpic++ $(FLAGS) -O3 $(SOURCE) -o tree_mpi.x

debug: main.cpp tree_mpi.cpp data_point.cpp
	mpic++ $(FLAGS) -D DEBUG -D OUTPUT $(SOURCE) -o tree_mpi.x

output: main.cpp tree_mpi.cpp data_point.cpp
	mpic++ $(FLAGS) -D OUTPUT $(SOURCE) -o tree_mpi.x

time: main.cpp tree_mpi.cpp data_point.cpp
	mpic++ $(FLAGS) -D TIME -O3 $(SOURCE) -o tree_mpi.x

leaks: main.cpp tree_mpi.cpp data_point.cpp
	mpic++ $(FLAGS) -static-libasan -fsanitize=address -g -D DEBUG $(SOURCE) -o tree_mpi.x

mpidebug: main.cpp tree_mpi.cpp data_point.cpp
	mpic++ $(FLAGS) -g -D DEBUG -D MPI_DEBUG $(SOURCE) -o tree_mpi.x

clean:
	rm *.x
	rm *.gch
