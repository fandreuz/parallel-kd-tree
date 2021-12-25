.DEFAULT_GOAL := output

compile: main.cpp tree_mpi.cpp data_point.h
	mpic++ -O3 -o tree_mpi data_point.h tree_mpi.cpp main.cpp

debug: main.cpp tree_mpi.cpp data_point.h
	mpic++ -D DEBUG -D OUTPUT -o tree_mpi data_point.h tree_mpi.cpp main.cpp

output: main.cpp tree_mpi.cpp data_point.h
	mpic++ -D OUTPUT -o tree_mpi data_point.h tree_mpi.cpp main.cpp

time: main.cpp tree_mpi.cpp data_point.h
	mpic++ -D TIME -O3 -o tree_mpi data_point.h tree_mpi.cpp main.cpp

leaks: main.cpp tree_mpi.cpp data_point.h
	mpic++ -static-libasan -fsanitize=address -g -D DEBUG -o tree_mpi data_point.h tree_mpi.cpp main.cpp

mpidebug: main.cpp tree_mpi.cpp data_point.h
	mpic++ -g -D DEBUG -D MPI_DEBUG -o tree_mpi data_point.h tree_mpi.cpp main.cpp
