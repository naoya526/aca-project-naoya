#!/bin/bash
mpiexec -n 1 --oversubscribe -hostfile ../hosts.txt ./../cnn_mpi 10
mpiexec -n 2 --oversubscribe -hostfile ../hosts.txt ./../cnn_mpi 20
mpiexec -n 4 --oversubscribe -hostfile ../hosts.txt ./../cnn_mpi 40
mpiexec -n 8 --oversubscribe -hostfile ../hosts.txt ./../cnn_mpi 80
