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

## Implementation

### Local Experiment
To enable seamless integration with frameworks like PyTorch, a Python wrapper is provided. For computational efficiency and parallelization, the core convolution operations are implemented in C with MPI. The Python code calls the optimized C functions via `ctypes`, combining ease of use with high performance.

### Execution(Local)

| # Processes | Execution Time | Description                                |
|------------:|----------------|--------------------------------------------|
| 1           | 0.602157 sec   | Serial execution (no parallelism)          |
| 2           | 0.266914 sec   | Parallel split (2 ranks, 2500 each)        |
| 4           | 0.134905 sec   | Parallel split (4 ranks, 625 each)         |


Example command:  
```bash
 mpicc -o cnn_mpi main.c conv2d.c -lm
 mpiexec -n 4 ./cnn_mpi
```
```bash
mpiexec -n 4 python3 wrapper_test.py
```

### Cloud Execution Results

| # Processes | Execution Time (max) | Local Batch per Rank | Start-End Indices per Rank                | Output Shape            |
|------------:|---------------------|----------------------|-------------------------------------------|------------------------|
| 1           | 0.735935 sec        | 10000                | 0-10000                                   | (10000, 1, 30, 30)     |
| 2           | 0.284851 sec        | 2500                 | 0-2500 (rank 0), 2500-5000 (rank 1)       | (10000, 1, 30, 30)     |
| 4           | 0.385728 sec        | 625                  | 0-625 (r0), 625-1250 (r1), 1250-1875 (r2), 1875-2500 (r3) | (10000, 1, 30, 30)     |

####    Example of Output Logs

```text
debian@light-node-0:~/aca-project-naoya/Local$ mpiexec --oversubscribe -np 4 -hostfile hosts.txt ./cnn_mpi
[MPI] rank=1, size=4, local_batch=625, start=625, end=1250
[MPI] rank=2, size=4, local_batch=625, start=1250, end=1875
[MPI] rank=0, size=4, local_batch=625, start=0, end=625
[MPI] rank=3, size=4, local_batch=625, start=1875, end=2500
Output shape: (10000, 1, 30, 30)
Executed time (max across ranks): 0.385728 sec
debian@light-node-0:~/aca-project-naoya/Local$ mpiexec --oversubscribe -np 2 -hostfile hosts.txt ./cnn_mpi
[MPI] rank=0, size=2, local_batch=2500, start=0, end=2500
[MPI] rank=1, size=2, local_batch=2500, start=2500, end=5000
Output shape: (10000, 1, 30, 30)
Executed time (max across ranks): 0.284851 sec
debian@light-node-0:~/aca-project-naoya/Local$ mpiexec --oversubscribe -np 1 -hostfile hosts.txt ./cnn_mpi
[MPI] rank=0, size=1, local_batch=10000, start=0, end=10000
Output shape: (10000, 1, 30, 30)
Executed time (max across ranks): 0.735935 sec
```

### Fat Node Execution Results

| # Processes | Execution Time (max) | Local Batch per Rank | Start-End Indices per Rank (examples)                | Output Shape            |
|------------:|---------------------|----------------------|------------------------------------------------------|------------------------|
| 1           | 0.537920 sec        | 10000                | 0-10000                                              | (10000, 1, 30, 30)     |
| 2           | 0.138425 sec        | 2500                 | 0-2500 (r0), 2500-5000 (r1)                          | (10000, 1, 30, 30)     |
| 4           | 0.045545 sec        | 625                  | 0-625 (r0), 625-1250 (r1), 1250-1875 (r2), 1875-2500 (r3) | (10000, 1, 30, 30)     |
| 8           | 0.024449 sec        | 156                  | 0-156 (r0), 156-312 (r1), ..., 1092-1248 (r7)        | (10000, 1, 30, 30)     |
| 16          | 0.005993 sec        | 39                   | 0-39 (r0), 39-78 (r1), ..., 585-624 (r15)            | (10000, 1, 30, 30)     |

*Note: Only representative start-end indices are shown for brevity.*


## Next Steps

- [ ] Launch a minimal Light Cluster using Terraform on GCP
- [ ] Validate MPI-based distributed runs in cloud
- [ ] Compare performance between local and cloud environments

## Author

- [Naoya Kumakura](https://github.com/naoya526)
