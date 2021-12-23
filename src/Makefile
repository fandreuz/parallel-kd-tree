.DEFAULT_GOAL := compile

compile: main.cpp tree_mpi.cpp data_point.h
	mpic++ -O3 -o tree_mpi data_point.h tree_mpi.cpp main.cpp

debug: main.cpp tree_mpi.cpp data_point.h
	mpic++ -static-libasan -fsanitize=address -g -D DEBUG -o tree_mpi data_point.h tree_mpi.cpp main.cpp

mpidebug: main.cpp tree_mpi.cpp data_point.h
	mpic++ -g -D DEBUG -D MPI_DEBUG -o tree_mpi data_point.h tree_mpi.cpp main.cpp
