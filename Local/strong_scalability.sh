#!/bin/bash
mpiexec -n 1 ./cnn_mpi16
mpiexec -n 2 ./cnn_mpi16
mpiexec -n 4 ./cnn_mpi16
mpiexec -n 8 ./cnn_mpi16
mpiexec -n 16 ./cnn_mpi16