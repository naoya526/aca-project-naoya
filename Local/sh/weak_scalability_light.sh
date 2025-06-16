#!/bin/bash
mpiexec -n 1 --oversubscribe -hostfile hosts.txt ./cnn_mpi1
mpiexec -n 2 --oversubscribe -hostfile hosts.txt ./cnn_mpi2
mpiexec -n 4 --oversubscribe -hostfile hosts.txt ./cnn_mpi4
mpiexec -n 8 --oversubscribe -hostfile hosts.txt ./cnn_mpi8
