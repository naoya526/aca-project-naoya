#!/bin/bash
mpiexec -n 1 ./cnn_mpi 10
mpiexec -n 2 ./cnn_mpi 20
mpiexec -n 4 ./cnn_mpi 40
mpiexec -n 8 ./cnn_mpi 80
mpiexec -n 16 ./cnn_mpi 160