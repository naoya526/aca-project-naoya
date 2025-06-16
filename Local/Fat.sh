#!/bin/bash
echo "Strong Scalability"
mpiexec -n 1 python3 main.py 80
mpiexec -n 2 python3 main.py 80
mpiexec -n 4 python3 main.py 80
mpiexec -n 8 python3 main.py 80
mpiexec -n 16 python3 main.py 80

echo "Weak Scalability"
mpiexec -n 1 python3 main.py 10
mpiexec -n 2 python3 main.py 20
mpiexec -n 4 python3 main.py 40
mpiexec -n 8 python3 main.py 80
mpiexec -n 16 python3 main.py 160