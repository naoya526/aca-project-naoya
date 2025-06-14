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

### 4.2 Local Execution Results
#### 4.2.1 Strong Scalanility

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
#### 4.2.2 Weak Scalanility

## 5 Experiment for Strong Scalability
### 5.1 Light Cluster
### 5.2 Fat Cluster
| # Processes | Execution Time (max) | Local Batch per Rank | Start-End Indices per Rank (examples)                                 | Output Shape            |
|------------:|---------------------|----------------------|-----------------------------------------------------------------------|------------------------|
| 1           | 84.838156 sec       | 160000               | 0-160000                                                              | (160000, 1, 30, 30)    |
| 2           | 24.410334 sec       | 80000                | 0-80000 (r0), 80000-160000 (r1)                                       | (160000, 1, 30, 30)    |
| 4           | 9.555175 sec        | 40000                | 0-40000 (r0), 40000-80000 (r1), 80000-120000 (r2), 120000-160000 (r3) | (160000, 1, 30, 30)    |
| 8           | 6.100789 sec        | 20000                | 0-20000 (r0), 20000-40000 (r1), ..., 140000-160000 (r7)               | (160000, 1, 30, 30)    |
| 16          | 2.986997 sec        | 10000                | 0-10000 (r0), 10000-20000 (r1), ..., 150000-160000 (r15)              | (160000, 1, 30, 30)    |
## 6 Experiment for Weak Scalability
### 6.1 Light Cluster


### 6.2 Fat Cluster
| # Processes | Execution Time (max) | Local Batch per Rank | Start-End Indices per Rank (examples)                | Output Shape            |
|------------:|---------------------|----------------------|------------------------------------------------------|------------------------|
| 1           | 5.002680 sec        | 100,000              | 0-100,000                                            | (100000, 1, 30, 30)    |
| 2           | 2.573401 sec        | 100,000              | 0-100,000 (r0), 100,000-200,000 (r1)                 | (200000, 1, 30, 30)    |
| 4           | 1.735116 sec        | 100,000              | 0-100,000 (r0), 100,000-200,000 (r1), 200,000-300,000 (r2), 300,000-400,000 (r3) | (400000, 1, 30, 30)    |
| 8           | 1.830275 sec        | 100,000              | 0-100,000 (r0), 100,000-200,000 (r1), ..., 700,000-800,000 (r7) | (800000, 1, 30, 30)    |
| 16          | 0.702247 sec        | 100,000              | 0-100,000 (r0), 100,000-200,000 (r1), ..., 1,500,000-1,600,000 (r15) | (1600000, 1, 30, 30)   |

## Next Steps

- [ ] Launch a minimal Light Cluster using Terraform on GCP
- [ ] Validate MPI-based distributed runs in cloud
- [ ] Compare performance between local and cloud environments

## Author

- [Naoya Kumakura](https://github.com/naoya526)
