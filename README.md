# Optimizing Convolutional Layers in Neural Networks with MPI

## Overview

This project explores parallelizing the forward and backward propagation of convolutional layers using OpenMPI, implemented in C and wrapped in Python. The primary goal is to improve performance through distributed computation across multiple processors.

## Current Status

-  Convolutional layer implementation (forward pass) in C
-  Python wrapper using `ctypes` or similar to interface C code
-  OpenMPI-based parallel execution confirmed on local machine
-  Docker-based development environment working
-  Tested local cluster performance using `mpiexec`
-  GCP infrastructure being prepared with Terraform (not yet deployed)

## Local Development Environment

- Language: C (core computation) + Python (wrapper + launcher)
- Parallelism: OpenMPI
- Local testing: Docker containers
- Build tool: `Makefile`
- Debugging: GDB (GNU Debugger)

## Execution Example (Local)

| # Processes | Execution Time | Description                                |
|------------:|----------------|--------------------------------------------|
| 1           | 0.525627 sec   | Serial execution (no parallelism)          |
| 2           | 0.201717 sec   | Parallel split (2 ranks, 2500 each)        |
| 4           | 0.070831 sec   | Parallel split (4 ranks, 625 each)         |

Example command:  
```bash
mpiexec -n 4 python3 wrapper_test.py
```

## Next Steps

- [ ] Launch a minimal Light Cluster using Terraform on GCP
- [ ] Validate MPI-based distributed runs in cloud
- [ ] Compare performance between local and cloud environments

## Author

- [Naoya Kumakura](https://github.com/naoya526)
