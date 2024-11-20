#!/bin/bash

# Experiment 2: Top n_probe Clusters (CPU Coarse Search)
EXECUTABLE="./IVF"
PARAMETER_CONFIGS=(
    "--n_probe=5 --use_cuda_coarse=false --use_cuda_fine=true --print_results=false"
    "--n_probe=20 --use_cuda_coarse=false --use_cuda_fine=true --print_results=false"
    "--n_probe=40 --use_cuda_coarse=false --use_cuda_fine=true --print_results=false"
)
NUM_RUNS=5

source ./run_multiple_configs.sh
