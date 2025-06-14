#!/bin/bash
sudo apt-get update
sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev
sudo apt-get install -y git
git clone https://github.com/naoya526/aca-project-naoya
mpiexec -n 1 ./cnn_mpi1
mpiexec -n 2 ./cnn_mpi2
mpiexec -n 4 ./cnn_mpi4
mpiexec -n 8 ./cnn_mpi8