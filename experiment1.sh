#!/bin/bash

# Experiment 1: GPU vs CPU
EXECUTABLE="./IVF"
PARAMETER_CONFIGS=(
    "--n_probe=5 --use_cuda_coarse=true --use_cuda_fine=true --print_results=false"
    "--n_probe=20 --use_cuda_coarse=true --use_cuda_fine=true --print_results=false"
    "--n_probe=40 --use_cuda_coarse=true --use_cuda_fine=true --print_results=false"
)
NUM_RUNS=5

source ./run_multiple_configs.sh
