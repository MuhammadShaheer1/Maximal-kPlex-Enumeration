#!/bin/bash
set -e

echo "Compiling CUDA project..."
nvcc -arch=sm_89 --maxrregcount=64 --extended-lambda -Xcompiler -fopenmp ./src/device_funcs.cu main.cu -o main
echo "Build finished: ./main"
