# Optimizing Convolutional Layers in Neural Networks with MPI

## 1. Overview

This project explores parallelizing the forward and backward propagation of convolutional layers using OpenMPI, implemented in C and wrapped in Python. The primary goal is to improve performance through distributed computation across multiple processors.


## 2. Current Status

-  Convolutional layer implementation (forward pass) in C
-  Python wrapper using `ctypes` or similar to interface C code
-  OpenMPI-based parallel execution confirmed on local machine
-  Docker-based development environment working
-  Tested local cluster performance using `mpiexec`
-  GCP infrastructure being prepared with Terraform
-  Validate MPI-based distributed runs in cloud
-  Compare performance between local and cloud environments

-  You can find the detailed report in the following PDF:

[ParallelComputing_NaoyaKumakura.pdf](./ParallelComputing_NaoyaKumakura.pdf)  
This document provides an explanation of the project, including background, methodology, experimental setup, results, and analysis.

## 3. Local Development Environment

- Language: C (core computation) + Python (wrapper + launcher + data loading(PIL image))
- Parallelism: OpenMPI
- Local testing: Ubuntu WSL2
- Analysing: mpiP
- Cloud Deployment: Terraform

## 4. Implementation

### 4.1 Local Experiment
To enable seamless integration with frameworks like PyTorch, a Python wrapper is provided. For computational efficiency and parallelization, the core convolution operations are implemented in C with MPI. The Python code calls the optimized C functions via `ctypes`, combining ease of use with high performance.

### 4.2 Local Execution Results

Example command:  
```bash
 mpicc -o cnn_mpi main.c conv.c -lm
 mpiexec -n 4 ./cnn_mpi
```
for using library of python image pil image, I also implemented on python. In this case, `conv.c` is wrapped with mpi.
```bash
mpicc -fPIC -shared -o libconv.so conv.c
mpiexec -n 4 python3 main.py
```

#### 4.2.1 Strong Scalanility
input:10000
| # Processes | Execution Time | Description                                |
|------------:|----------------|--------------------------------------------|
| 1           | 0.602157 sec   | Serial execution (no parallelism)          |
| 2           | 0.266914 sec   | Parallel split (2 ranks, 2500 each)        |
| 4           | 0.134905 sec   | Parallel split (4 ranks, 625 each)         |

#### 4.2.2 Weak Scalability
input:10000 for each processor
| # Processes | Execution Time | Description                                |
|------------:|----------------|--------------------------------------------|
| 1           | 0.610445 sec   | Serial execution (no parallelism)          |
| 2           | 0.536835 sec   | Parallel split (2 ranks, 2500 each)        |
| 4           | 0.657419 sec   | Parallel split (4 ranks, )         |

## 5 Experiment for Strong Scalability
### 5.1 Light Cluster
### 5.2 Fat Cluster
| # Processes | Execution Time (max) | Memory Usage per Rank (MB)                                 | Total Memory Usage (MB) | Output Shape             |
|------------:|---------------------|-------------------------------------------------------------|------------------------|--------------------------|
| 1           | 44.02 sec           | [1034.58]                                                   | 1034.58                | (80, 3, 1022, 1022)      |
| 2           | 11.10 sec           | [1034.45, 72.77]                                            | 1107.22                | (80, 3, 1022, 1022)      |
| 4           | 3.04 sec            | [1034.35, 72.91, 73.28, 72.74]                              | 1253.28                | (80, 3, 1022, 1022)      |
| 8           | 0.88 sec            | [1034.45, 73.06, 72.88, 72.95, 70.81, 70.89, 72.99, 73.01]  | 1541.04                | (80, 3, 1022, 1022)      |



## 6 Experiment for Weak Scalability
### 6.1 Light Cluster
### 6.2 Fat Cluster
| # Processes | Execution Time (max) | Memory Usage per Rank (MB)                                 | Total Memory Usage (MB) | Output Shape             |
|------------:|---------------------|-------------------------------------------------------------|------------------------|--------------------------|
| 1           | 5.51 sec            | [194.39]                                                    | 194.39                 | (10, 3, 1022, 1022)      |
| 2           | 2.78 sec            | [314.57, 72.88]                                             | 387.45                 | (20, 3, 1022, 1022)      |
| 4           | 1.24 sec            | [556.62, 72.77, 72.84, 72.96]                               | 775.20                 | (40, 3, 1022, 1022)      |
| 8           | 0.87 sec            | [1034.52, 72.79, 72.71, 72.90, 70.70, 72.70, 72.93, 72.94]  | 1542.19                | (80, 3, 1022, 1022)      |


# (Appendix: For small image and Huge number of Batch Size)
## Experiment for Strong Scalability
### Light Cluster
| # Processes | Execution Time (max) | Local Batch per Rank | Start-End Indices per Rank (examples)                                           | Output Shape           |
|------------:|---------------------|----------------------|----------------------------------------------------------------------------------|------------------------|
| 1           | 13.484401 sec       | 160,000              | 0-160,000                                                                        | (160,000, 1, 30, 30)   |
| 2           | 9.518927 sec        | 80,000               | 0-80,000 (r0), 80,000-160,000 (r1)                                               | (160,000, 1, 30, 30)   |
| 4           | 7.307634 sec        | 40,000               | 0-40,000 (r0), 40,000-80,000 (r1), 80,000-120,000 (r2), 120,000-160,000 (r3)     | (160,000, 1, 30, 30)   |
| 8           | 6.058151 sec        | 20,000               | 0-20,000 (r0), 20,000-40,000 (r1), 40,000-60,000 (r2), ..., 140,000-160,000(r7)  | (160,000, 1, 30, 30)   |

### Fat Cluster
| # Processes | Execution Time (max) | Local Batch per Rank | Start-End Indices per Rank (examples)                                 | Output Shape            |
|------------:|---------------------|----------------------|-----------------------------------------------------------------------|------------------------|
| 1           | 84.838156 sec       | 160000               | 0-160000                                                              | (160000, 1, 30, 30)    |
| 2           | 24.410334 sec       | 80000                | 0-80000 (r0), 80000-160000 (r1)                                       | (160000, 1, 30, 30)    |
| 4           | 9.555175 sec        | 40000                | 0-40000 (r0), 40000-80000 (r1), 80000-120000 (r2), 120000-160000 (r3) | (160000, 1, 30, 30)    |
| 8           | 6.100789 sec        | 20000                | 0-20000 (r0), 20000-40000 (r1), ..., 140000-160000 (r7)               | (160000, 1, 30, 30)    |
| 16          | 2.986997 sec        | 10000                | 0-10000 (r0), 10000-20000 (r1), ..., 150000-160000 (r15)              | (160000, 1, 30, 30)    |


## Experiment for Weak Scalability
### Light Cluster
| # Processes | Execution Time (max) | Local Batch per Rank | Start-End Indices per Rank (examples)                                     | Output Shape           |
|------------:|---------------------|----------------------|----------------------------------------------------------------------------|------------------------|
| 1           | 0.744552 sec        | 10,000               | 0-10,000                                                                   | (10,000, 1, 30, 30)    |
| 2           | 1.056537 sec        | 5,000                | 0-10,000 (r0), 10,000-20,000 (r1)                                          | (20,000, 1, 30, 30)    |
| 4           | 2.039511 sec        | 2,500                | 0-10,000 (r0), 10,000-20,000 (r1), 20,000-30,000 (r2), 30,000-40,000 (r3)  | (40,000, 1, 30, 30)    |
| 8           | 3.729983 sec        | 1,250                | 0-10,000 (r0), 10,000-20,000 (r1), ..., 70,000-80,000 (r7)                 | (80,000, 1, 30, 30)    |

### Fat Cluster
| # Processes | Execution Time (max) | Local Batch per Rank | Start-End Indices per Rank (examples)                | Output Shape            |
|------------:|---------------------|----------------------|------------------------------------------------------|------------------------|
| 1           | 5.002680 sec        | 100,000              | 0-100,000                                            | (100000, 1, 30, 30)    |
| 2           | 2.573401 sec        | 100,000              | 0-100,000 (r0), 100,000-200,000 (r1)                 | (200000, 1, 30, 30)    |
| 4           | 1.735116 sec        | 100,000              | 0-100,000 (r0), 100,000-200,000 (r1), 200,000-300,000 (r2), 300,000-400,000 (r3) | (400000, 1, 30, 30)    |
| 8           | 1.830275 sec        | 100,000              | 0-100,000 (r0), 100,000-200,000 (r1), ..., 700,000-800,000 (r7) | (800000, 1, 30, 30)    |
| 16          | 0.702247 sec        | 100,000              | 0-100,000 (r0), 100,000-200,000 (r1), ..., 1,500,000-1,600,000 (r15) | (1600000, 1, 30, 30)   |


## Author
- [Naoya Kumakura](https://github.com/naoya526)
