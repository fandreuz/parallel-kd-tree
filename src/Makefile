.DEFAULT_GOAL := compile

compile: main.cpp tree_mpi.cpp
	mpic++ -O3 -o tree_mpi tree_mpi.cpp main.cpp

debug: main.cpp tree_mpi.cpp
	mpic++ -O3 -g -D DEBUG -o tree_mpi tree_mpi.cpp main.cpp
