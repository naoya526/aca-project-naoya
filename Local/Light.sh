#!/bin/bash
echo "Strong Scalability"
mpiexec -hostfile hosts.txt -n 1 python3 main.py 40
mpiexec -hostfile hosts.txt -n 2 python3 main.py 40
mpiexec -hostfile hosts.txt -n 4 python3 main.py 40
mpiexec -hostfile hosts.txt -n 8 python3 main.py 40

echo "Weak Scalability"
mpiexec -hostfile hosts.txt -n 1 python3 main.py 5
mpiexec -hostfile hosts.txt -n 2 python3 main.py 10
mpiexec -hostfile hosts.txt -n 4 python3 main.py 20
mpiexec -hostfile hosts.txt -n 8 python3 main.py 40

echo "if possible, execute this"
echo "Strong Scalability"
mpiexec -hostfile hosts.txt -n 1 python3 main.py 80
mpiexec -hostfile hosts.txt -n 2 python3 main.py 80
mpiexec -hostfile hosts.txt -n 4 python3 main.py 80
mpiexec -hostfile hosts.txt -n 8 python3 main.py 80

echo "Weak Scalability"
mpiexec -hostfile hosts.txt -n 1 python3 main.py 10
mpiexec -hostfile hosts.txt -n 2 python3 main.py 20
mpiexec -hostfile hosts.txt -n 4 python3 main.py 40
mpiexec -hostfile hosts.txt -n 8 python3 main.py 80