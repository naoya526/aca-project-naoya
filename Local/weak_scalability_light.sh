#!/bin/bash
mpiexec -n 1 -hostfile hosts.txt ./cnn_mpi1
mpiexec -n 2 -hostfile hosts.txt ./cnn_mpi2
mpiexec -n 4 -hostfile hosts.txt ./cnn_mpi4
mpiexec -n 8 -hostfile hosts.txt ./cnn_mpi8
