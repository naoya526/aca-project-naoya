#!/bin/bash
mpiexec -n 1 ./cnn_mpi
mpiexec -n 2 ./cnn_mpi
mpiexec -n 4 ./cnn_mpi
mpiexec -n 8 ./cnn_mpi
mpiexec -n 16 ./cnn_mpi