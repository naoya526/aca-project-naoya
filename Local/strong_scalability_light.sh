#!/bin/bash
mpiexec -n 1 -hostfile hosts.txt ./cnn_mpi16
mpiexec -n 2 -hostfile hosts.txt ./cnn_mpi16
mpiexec -n 4 -hostfile hosts.txt ./cnn_mpi16
mpiexec -n 8 -hostfile hosts.txt ./cnn_mpi16
