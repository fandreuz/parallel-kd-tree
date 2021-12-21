.DEFAULT_GOAL := compile

compile: main.cpp tree_mpi.cpp
	mpic++ -O3 -o tree_mpi tree_mpi.cpp main.cpp

debug: main.cpp tree_mpi.cpp
	mpic++ -O3 -g -D DEBUG -o tree_mpi tree_mpi.cpp main.cpp

mpidebug: main.cpp tree_mpi.cpp
	export MPI_DEBUG_RANK="0"
	mpic++ -O3 -g -D DEBUG -D MPI_DEBUG -o tree_mpi tree_mpi.cpp main.cpp
