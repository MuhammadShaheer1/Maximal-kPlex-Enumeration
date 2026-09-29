# CudaPlex: GPU-Accelerated Maximal k-Plex Enumeration

This repository contains **CudaPlex**, a GPU-based algorithm for enumerating all maximal k-plexes of size at least q in large graphs. Unlike prior maximal k-plex enumeration methods that target CPUs, CudaPlex is designed specifically for GPUs and addresses the core challenges of recursion, irregular memory access, workload imbalance, and limited device memory.

The method is described in the accompanying research paper, *GPU-Accelerated Maximal k-Plex Enumeration*.

## Overview

A k-plex is a relaxed clique model in which each vertex may miss up to k-1 edges inside the subgraph. A maximal k-plex is one that cannot be extended by adding another vertex while preserving the k-plex property.

CudaPlex is, to the best of our knowledge, the **first GPU-oriented implementation for maximal k-plex enumeration**. The system reformulates recursive CPU-style search into a GPU-friendly execution model with:

- **Iterative stack-based search** instead of deep recursion
- **Fine-grained task scheduling** to reduce warp idling and improve load balance
- **Pause-and-resume memory management** to control task explosion on large graphs
- **GPU-optimized pruning** with compact state representations
- Optional **(q-2k)-truss-based preprocessing** for additional pruning on small and medium graphs

Experiments reported in the paper show up to **23.7x speedup** over a state-of-the-art parallel CPU baseline.

## Repository Structure

```text
.
├── dataset/          # Sample graph datasets in binary format
├── inc/              # Header files and host-side utilities
├── src/              # CUDA device-side implementation
├── compile.sh        # Simple NVCC build script
├── main.cu           # Program entry point
└── README.md
```

## Requirements

This project targets Linux and NVIDIA GPUs.

### Tested environment

- Ubuntu 24.04.3 LTS
- CUDA Toolkit 12.8
- NVIDIA RTX 6000 Ada Generation GPU
- AMD Ryzen Threadripper PRO 7985WX CPU (for preprocessing / baseline experiments)

### Build dependencies

- `nvcc`
- GNU C/C++ toolchain
- OpenMP support
- Bash

> **Note:** the provided `compile.sh` currently compiles with `-arch=sm_89`. Update this flag if your GPU uses a different compute capability.

## Build

### Build with the provided script

```bash
chmod +x compile.sh
./compile.sh
```

This builds the executable:

```bash
./main
```

## Usage

The current command-line interface is based on the repository entry point in `main.cu`.

```bash
./main <dataset> -k <k> -q <q> -t <threshold> -tr <truss>
```

### Parameters

- `<dataset>`: input graph file
- `-k`: maximum number of non-neighbors allowed per vertex inside a \(k\)-plex
- `-q`: minimum size threshold for enumerated maximal \(k\)-plexes
- `-t`: threshold set in the pause-and-resume strategy
- `-tr`: Enables (1) or disables (0) k-Truss filtering.

### Example

```bash
./main ./dataset/soc-epinions.bin -k 4 -q 30 -t 0.25 -tr 0
```

> Depending on your local version of the code, you may want to run `./main` without arguments first and verify the printed usage string.

## Input Data

The repository includes sample preprocessed graph datasets in `dataset/`.

The repository includes files such as:

- `com-dblp.bin`
- `email-euall.bin`
- `slashdot.bin`
- `soc-epinions.bin`

These are binary graph inputs used for enumeration experiments. Other datasets can be obtained from the SNAP and LAW repository.

## Method Summary

CudaPlex combines CPU preprocessing with GPU search.

1. **Preprocessing on CPU**
   - Reduce the graph to its (q-k)-core
   - Compute a degeneracy ordering
   - Optionally apply k-truss pruning for additional graph reduction

2. **GPU decomposition**
   - For each root vertex, construct disjoint sets \(P\), \(C_1\), \(C_2\), and \(X\)
   - \(P\): current partial solution
   - \(C_1\): one-hop candidates
   - \(C_2\): two-hop candidates
   - \(X\): excluded vertices

3. **Iterative k-search**
   - Replace recursive exploration with an explicit stack-based search over \(C_2\)

4. **Task-based branch-and-bound**
   - Convert Branch calls into lightweight tasks
   - Use one global queue and two auxiliary queues for load balancing across warps

5. **Pause-and-resume control**
   - Pause task generation when the global queue occupancy reaches a threshold
   - Process queued tasks and then resume from the stored search state

This design avoids deep recursion on the GPU, improves occupancy, and bounds memory usage even when the search generates extremely large numbers of tasks.

## Experimental Highlights

The paper evaluates CudaPlex on 10 real-world datasets.

Selected results reported in the paper:

| Dataset | k | q | # Maximal k-Plexes | CudaPlex (s) | Parallel CPU (s) |
|---|---:|---:|---:|---:|---:|
| soc-epinions | 4 | 30 | 13,172,906 | 4.13 | 64.33 |
| soc-slashdot | 4 | 30 | 1,047,289,095 | 99.65 | 1732.28 |
| email-euall | 3 | 12 | 32,639,016 | 1.32 | 6.60 |
| com-dblp | 5 | 25 | 3,804,584,758 | 1132.86 | Failed |

Overall, the method achieves **2.3x to 23.7x speedup** over the parallel CPU baseline, depending on the dataset and parameter setting.

## Notes on Performance

The paper shows that performance gains come from three main implementation ideas:

- replacing naive root-based GPU execution with a **fine-grained task scheduler**,
- using a **pause-and-resume threshold** to avoid task-queue overflow,
- and applying **additional graph pruning** when beneficial.

The scheduler is especially important: the paper reports substantial speedups over simpler GPU baselines that do not balance work effectively.
