#!/bin/bash
mpiexec --oversubscribe -np 1 -hostfile ../hosts.txt ./../cnn_mpi 80
mpiexec --oversubscribe -np 2 -hostfile ../hosts.txt ./../cnn_mpi 80
mpiexec --oversubscribe -np 4 -hostfile ../hosts.txt ./../cnn_mpi 80
mpiexec --oversubscribe -np 8 -hostfile ../hosts.txt ./../cnn_mpi 80
