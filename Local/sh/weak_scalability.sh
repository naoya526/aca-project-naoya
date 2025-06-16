#!/bin/bash
mpiexec -n 1 ./cnn_mpi1
mpiexec -n 2 ./cnn_mpi2
mpiexec -n 4 ./cnn_mpi4
mpiexec -n 8 ./cnn_mpi8
mpiexec -n 16 ./cnn_mpi16