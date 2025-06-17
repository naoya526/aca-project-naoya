#!/bin/bash
mpiexec -n 1 --oversubscribe -hostfile ../hosts.txt ./../cnn_mpi
mpiexec -n 2 --oversubscribe -hostfile ../hosts.txt ./../cnn_mpi
mpiexec -n 4 --oversubscribe -hostfile ../hosts.txt ./../cnn_mpi
mpiexec -n 8 --oversubscribe -hostfile ../hosts.txt ./../cnn_mpi
