# Optimizing Convolutional Layers in Neural Networks with MPI

## 1. Overview

This project explores parallelizing the forward and backward propagation of convolutional layers using OpenMPI, implemented in C and wrapped in Python. The primary goal is to improve performance through distributed computation across multiple processors.

## 2. Current Status

-  Convolutional layer implementation (forward pass) in C
-  Python wrapper using `ctypes` or similar to interface C code
-  OpenMPI-based parallel execution confirmed on local machine
-  Docker-based development environment working
-  Tested local cluster performance using `mpiexec`
-  GCP infrastructure being prepared with Terraform (not yet deployed)

## 3. Local Development Environment

- Language: C (core computation) + Python (wrapper + launcher)
- Parallelism: OpenMPI
- Local testing: Docker containers
- Build tool: `Makefile`
- Debugging: GDB (GNU Debugger)

## 4. Implementation

### 4.1 Local Experiment
To enable seamless integration with frameworks like PyTorch, a Python wrapper is provided. For computational efficiency and parallelization, the core convolution operations are implemented in C with MPI. The Python code calls the optimized C functions via `ctypes`, combining ease of use with high performance.

### 4.2 Local Execution Results(Strong Scalanility)

| # Processes | Execution Time | Description                                |
|------------:|----------------|--------------------------------------------|
| 1           | 0.602157 sec   | Serial execution (no parallelism)          |
| 2           | 0.266914 sec   | Parallel split (2 ranks, 2500 each)        |
| 4           | 0.134905 sec   | Parallel split (4 ranks, 625 each)         |


Example command:  
```bash
 mpicc -o cnn_mpi main.c conv.c -lm
 mpiexec -n 4 ./cnn_mpi
```
```bash
mpiexec -n 4 python3 wrapper_test.py
```

## 5 Experiment for Strong Scalability
### 5.1 Light Cluster
| # Processes | Execution Time (max) | Local Batch per Rank | Start-End Indices per Rank                | Output Shape            |
|------------:|---------------------|----------------------|-------------------------------------------|------------------------|
| 1           | 0.735935 sec        | 10000                | 0-10000                                   | (10000, 1, 30, 30)     |
| 2           | 0.284851 sec        | 2500                 | 0-2500 (rank 0), 2500-5000 (rank 1)       | (10000, 1, 30, 30)     |
| 4           | 0.385728 sec        | 625                  | 0-625 (r0), 625-1250 (r1), 1250-1875 (r2), 1875-2500 (r3) | (10000, 1, 30, 30)     |


### 5.2 Fat Cluster

| # Processes | Execution Time (max) | Local Batch per Rank | Start-End Indices per Rank (examples)                | Output Shape            |
|------------:|---------------------|----------------------|------------------------------------------------------|------------------------|
| 1           | 0.537920 sec        | 10000                | 0-10000                                              | (10000, 1, 30, 30)     |
| 2           | 0.138425 sec        | 2500                 | 0-2500 (r0), 2500-5000 (r1)                          | (10000, 1, 30, 30)     |
| 4           | 0.045545 sec        | 625                  | 0-625 (r0), 625-1250 (r1), 1250-1875 (r2), 1875-2500 (r3) | (10000, 1, 30, 30)     |
| 8           | 0.024449 sec        | 156                  | 0-156 (r0), 156-312 (r1), ..., 1092-1248 (r7)        | (10000, 1, 30, 30)     |
| 16          | 0.005993 sec        | 39                   | 0-39 (r0), 39-78 (r1), ..., 585-624 (r15)            | (10000, 1, 30, 30)     |

mpiexec -n 1 ./cnn_mpi16
[MPI] rank=0, size=1, local_batch=1600000, start=0, end=1600000
Output shape: (1600000, 1, 30, 30)
Executed time (max across ranks): 80.551997 sec
mpiexec -n 2 ./cnn_mpi16
[MPI] rank=0, size=2, local_batch=400000, start=0, end=400000
[MPI] rank=1, size=2, local_batch=400000, start=400000, end=800000
Output shape: (1600000, 1, 30, 30)
Executed time (max across ranks): 20.690116 sec
mpiexec -n 4 ./cnn_mpi16
[MPI] rank=1, size=4, local_batch=100000, start=100000, end=200000
[MPI] rank=2, size=4, local_batch=100000, start=200000, end=300000
[MPI] rank=0, size=4, local_batch=100000, start=0, end=100000
[MPI] rank=3, size=4, local_batch=100000, start=300000, end=400000
Output shape: (1600000, 1, 30, 30)
Executed time (max across ranks): 6.708439 sec

## 6 Experiment for Weak Scalability
### 6.1 Light Cluster

### 6.2 Fat Cluster

| # Processes | Local Batch per Rank | Total Batch Size | Output Shape            | Execution Time (max) |
|------------:|---------------------|------------------|------------------------|----------------------|
| 1           | 100,000             | 100,000          | (100000, 1, 30, 30)    | 5.002680 sec         |
| 2           | 100,000             | 200,000          | (200000, 1, 30, 30)    | 2.573401 sec         |
| 4           | 100,000             | 400,000          | (400000, 1, 30, 30)    | 1.735116 sec         |
| 8           | 100,000             | 800,000          | (800000, 1, 30, 30)    | 1.830275 sec         |
| 16          | 100,000             | 1,600,000        | (1600000, 1, 30, 30)   | 0.702247 sec         |

| # Processes | Execution Time (max) | Local Batch per Rank | Start-End Indices per Rank (examples)                | Output Shape            |
|------------:|---------------------|----------------------|------------------------------------------------------|------------------------|
| 1           | 5.002680 sec        | 100,000              | 0-100,000                                            | (100000, 1, 30, 30)    |
| 2           | 2.573401 sec        | 100,000              | 0-100,000 (r0), 100,000-200,000 (r1)                 | (200000, 1, 30, 30)    |
| 4           | 1.735116 sec        | 100,000              | 0-100,000 (r0), 100,000-200,000 (r1), 200,000-300,000 (r2), 300,000-400,000 (r3) | (400000, 1, 30, 30)    |
| 8           | 1.830275 sec        | 100,000              | 0-100,000 (r0), 100,000-200,000 (r1), ..., 700,000-800,000 (r7) | (800000, 1, 30, 30)    |
| 16          | 0.702247 sec        | 100,000              | 0-100,000 (r0), 100,000-200,000 (r1), ..., 1,500,000-1,600,000 (r15) | (1600000, 1, 30, 30)   |


mpiexec -n 1 ./cnn_mpi1
mpiexec -n 2 ./cnn_mpi2
mpiexec -n 4 ./cnn_mpi4
mpiexec -n 8 ./cnn_mpi8
mpiexec -n 16 ./cnn_mpi16
[MPI] rank=0, size=1, local_batch=100000, start=0, end=100000
Output shape: (100000, 1, 30, 30)
Executed time (max across ranks): 5.002680 sec
[MPI] rank=1, size=2, local_batch=50000, start=50000, end=100000
[MPI] rank=0, size=2, local_batch=50000, start=0, end=50000
Output shape: (200000, 1, 30, 30)
Executed time (max across ranks): 2.573401 sec
[MPI] rank=1, size=4, local_batch=25000, start=25000, end=50000
[MPI] rank=2, size=4, local_batch=25000, start=50000, end=75000
[MPI] rank=3, size=4, local_batch=25000, start=75000, end=100000
[MPI] rank=0, size=4, local_batch=25000, start=0, end=25000
Output shape: (400000, 1, 30, 30)
Executed time (max across ranks): 1.735116 sec
[MPI] rank=1, size=8, local_batch=12500, start=12500, end=25000
[MPI] rank=2, size=8, local_batch=12500, start=25000, end=37500
[MPI] rank=3, size=8, local_batch=12500, start=37500, end=50000
[MPI] rank=4, size=8, local_batch=12500, start=50000, end=62500
[MPI] rank=5, size=8, local_batch=12500, start=62500, end=75000
[MPI] rank=6, size=8, local_batch=12500, start=75000, end=87500
[MPI] rank=0, size=8, local_batch=12500, start=0, end=12500
[MPI] rank=7, size=8, local_batch=12500, start=87500, end=100000
Output shape: (800000, 1, 30, 30)
Executed time (max across ranks): 1.830275 sec
[MPI] rank=12, size=16, local_batch=6250, start=75000, end=81250
[MPI] rank=6, size=16, local_batch=6250, start=37500, end=43750
[MPI] rank=4, size=16, local_batch=6250, start=25000, end=31250
[MPI] rank=15, size=16, local_batch=6250, start=93750, end=100000
[MPI] rank=14, size=16, local_batch=6250, start=87500, end=93750
[MPI] rank=10, size=16, local_batch=6250, start=62500, end=68750
[MPI] rank=13, size=16, local_batch=6250, start=81250, end=87500
[MPI] rank=2, size=16, local_batch=6250, start=12500, end=18750
[MPI] rank=8, size=16, local_batch=6250, start=50000, end=56250
[MPI] rank=11, size=16, local_batch=6250, start=68750, end=75000
[MPI] rank=1, size=16, local_batch=6250, start=6250, end=12500
[MPI] rank=5, size=16, local_batch=6250, start=31250, end=37500
[MPI] rank=7, size=16, local_batch=6250, start=43750, end=50000
[MPI] rank=3, size=16, local_batch=6250, start=18750, end=25000
[MPI] rank=9, size=16, local_batch=6250, start=56250, end=62500
[MPI] rank=0, size=16, local_batch=6250, start=0, end=6250
Output shape: (1600000, 1, 30, 30)
Executed time (max across ranks): 0.702247 sec

###
debian@fat-node-0:~/aca-project-naoya/Local/aca-project-naoya/Local$ mpiexec -n 16 ./cnn_mpi8
[MPI] rank=2, size=16, local_batch=3125, start=6250, end=9375
[MPI] rank=14, size=16, local_batch=3125, start=43750, end=46875
[MPI] rank=4, size=16, local_batch=3125, start=12500, end=15625
[MPI] rank=6, size=16, local_batch=3125, start=18750, end=21875
[MPI] rank=8, size=16, local_batch=3125, start=25000, end=28125
[MPI] rank=10, size=16, local_batch=3125, start=31250, end=34375
[MPI] rank=12, size=16, local_batch=3125, start=37500, end=40625
[MPI] rank=1, size=16, local_batch=3125, start=3125, end=6250
[MPI] rank=7, size=16, local_batch=3125, start=21875, end=25000
[MPI] rank=9, size=16, local_batch=3125, start=28125, end=31250
[MPI] rank=15, size=16, local_batch=3125, start=46875, end=50000
[MPI] rank=11, size=16, local_batch=3125, start=34375, end=37500
[MPI] rank=5, size=16, local_batch=3125, start=15625, end=18750
[MPI] rank=3, size=16, local_batch=3125, start=9375, end=12500
[MPI] rank=13, size=16, local_batch=3125, start=40625, end=43750
[MPI] rank=0, size=16, local_batch=3125, start=0, end=3125
Output shape: (800000, 1, 30, 30)
Executed time (max across ranks): 0.364449 sec
debian@fat-node-0:~/aca-project-naoya/Local/aca-project-naoya/Local$ mpiexec -n 8 ./cnn_mpi8
[MPI] rank=1, size=8, local_batch=12500, start=12500, end=25000
[MPI] rank=2, size=8, local_batch=12500, start=25000, end=37500
[MPI] rank=3, size=8, local_batch=12500, start=37500, end=50000
[MPI] rank=4, size=8, local_batch=12500, start=50000, end=62500
[MPI] rank=5, size=8, local_batch=12500, start=62500, end=75000
[MPI] rank=6, size=8, local_batch=12500, start=75000, end=87500
[MPI] rank=7, size=8, local_batch=12500, start=87500, end=100000
[MPI] rank=0, size=8, local_batch=12500, start=0, end=12500
Output shape: (800000, 1, 30, 30)
Executed time (max across ranks): 1.822475 sec
debian@fat-node-0:~/aca-project-naoya/Local/aca-project-naoya/Local$ mpiexec -n 4 ./cnn_mpi8
[MPI] rank=1, size=4, local_batch=50000, start=50000, end=100000
[MPI] rank=2, size=4, local_batch=50000, start=100000, end=150000
[MPI] rank=3, size=4, local_batch=50000, start=150000, end=200000
[MPI] rank=0, size=4, local_batch=50000, start=0, end=50000
Output shape: (800000, 1, 30, 30)
Executed time (max across ranks): 3.456056 sec
debian@fat-node-0:~/aca-project-naoya/Local/aca-project-naoya/Local$ mpiexec -n 2 ./cnn_mpi8
[MPI] rank=0, size=2, local_batch=200000, start=0, end=200000
[MPI] rank=1, size=2, local_batch=200000, start=200000, end=400000
Output shape: (800000, 1, 30, 30)
Executed time (max across ranks): 10.358678 sec
debian@fat-node-0:~/aca-project-naoya/Local/aca-project-naoya/Local$ mpiexec -n 1 ./cnn_mpi8
[MPI] rank=0, size=1, local_batch=800000, start=0, end=800000
Output shape: (800000, 1, 30, 30)
Executed time (max across ranks): 40.021159 sec

## Next Steps

- [ ] Launch a minimal Light Cluster using Terraform on GCP
- [ ] Validate MPI-based distributed runs in cloud
- [ ] Compare performance between local and cloud environments

## Author

- [Naoya Kumakura](https://github.com/naoya526)
