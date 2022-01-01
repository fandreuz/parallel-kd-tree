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

# eliminate default suffixes
.SUFFIXES:
SUFFIXES=

.SUFFIXES: .cpp .x .h

compile: main.cpp tree_mpi.cpp data_point.cpp
	mpic++ -std=c++14 -Wall -Wextra -O3 data_point.h tree_mpi.cpp main.cpp -o tree_mpi.x

debug: main.cpp tree_mpi.cpp data_point.cpp
	mpic++ -std=c++14 -Wall -Wextra -D DEBUG -D OUTPUT data_point.h tree_mpi.cpp main.cpp -o tree_mpi.x

output: main.cpp tree_mpi.cpp data_point.cpp
	mpic++ -std=c++14 -Wall -Wextra -D OUTPUT data_point.cpp tree_mpi.cpp main.cpp -o tree_mpi.x

time: main.cpp tree_mpi.cpp data_point.cpp
	mpic++ -std=c++14 -Wall -Wextra -D TIME -O3 data_point.cpp tree_mpi.cpp main.cpp -o tree_mpi.x

leaks: main.cpp tree_mpi.cpp data_point.cpp
	mpic++ -std=c++14 -Wall -Wextra -static-libasan -fsanitize=address -g -D DEBUG data_point.cpp tree_mpi.cpp main.cpp -o tree_mpi.x

mpidebug: main.cpp tree_mpi.cpp data_point.cpp
	mpic++ -std=c++14 -Wall -Wextra -g -D DEBUG -D MPI_DEBUG data_point.cpp tree_mpi.cpp main.cpp -o tree_mpi.x

clean:
	rm *.x
	rm *.gch
