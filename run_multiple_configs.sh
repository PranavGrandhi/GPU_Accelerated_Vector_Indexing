#!/bin/bash

# Script to run the executable multiple times across different parameter configurations
# and compute the average Search Time for each configuration.

# ----------------------------
# Configuration Section
# ----------------------------

# Default number of runs per configuration
DEFAULT_NUM_RUNS=5

# Define the executable
EXECUTABLE="./IVF"  # Replace with your actual executable name/path

# Define parameter configurations
# Each configuration is a string with parameters separated by spaces
# Format: "<n_probe> <Atomic|NonAtomic> <SequentialFineSearch> <--use_cuda_coarse=...> <--use_cuda_fine=...>"

# "--n_probe=5 --mode=Atomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=5 --mode=Atomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=5 --mode=NonAtomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=5 --mode=NonAtomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=20 --mode=Atomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=20 --mode=Atomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=20 --mode=NonAtomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=20 --mode=NonAtomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=40 --mode=Atomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=40 --mode=Atomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=40 --mode=NonAtomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=40 --mode=NonAtomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=80 --mode=Atomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=80 --mode=NonAtomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=40 --mode=NonAtomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=128 --print_results=false"
    # "--n_probe=40 --mode=NonAtomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=40 --mode=NonAtomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=512 --print_results=false"
    # "--n_probe=40 --mode=NonAtomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=1024 --print_results=false"
    # "--n_probe=40 --mode=NonAtomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=128 --print_results=false"
    # "--n_probe=40 --mode=NonAtomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=256 --print_results=false"
    # "--n_probe=40 --mode=NonAtomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=512 --print_results=false"
    # "--n_probe=40 --mode=NonAtomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=1024 --print_results=false"

PARAMETER_CONFIGS=(
    "--n_probe=80 --mode=Atomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=32 --print_results=false"
    "--n_probe=80 --mode=Atomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=32 --print_results=false"
    "--n_probe=80 --mode=Atomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=1024 --print_results=false"
    "--n_probe=80 --mode=Atomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=1024 --print_results=false"
    "--n_probe=80 --mode=NonAtomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=32 --print_results=false"
    "--n_probe=80 --mode=NonAtomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=32 --print_results=false"
    "--n_probe=80 --mode=NonAtomic --sequential_fine_search=true --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=1024 --print_results=false"
    "--n_probe=80 --mode=NonAtomic --sequential_fine_search=false --use_cuda_coarse=false --use_cuda_fine=true --threadsperBlock=1024 --print_results=false"
)

# ----------------------------
# End of Configuration Section
# ----------------------------

# Function to display usage
usage() {
    echo "Usage: $0 [<num_runs>]"
    echo " - <num_runs>: (Optional) Number of runs per configuration. Defaults to $DEFAULT_NUM_RUNS."
    echo "Example:"
    echo "   $0 10"
    echo "   Runs each configuration 10 times."
}

# Parse optional argument for number of runs
if [ $# -gt 1 ]; then
    echo "Error: Too many arguments."
    usage
    exit 1
fi

if [ $# -eq 1 ]; then
    if [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
        NUM_RUNS=$1
    else
        echo "Error: <num_runs> must be a positive integer."
        usage
        exit 1
    fi
else
    NUM_RUNS=$DEFAULT_NUM_RUNS
fi

# Check if the executable exists and is executable
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: Executable '$EXECUTABLE' not found or not executable."
    exit 1
fi

# Temporary file to store the output of each run
temp_file=$(mktemp)

# Iterate over each parameter configuration
for CONFIG in "${PARAMETER_CONFIGS[@]}"; do
    echo "=============================================="
    echo "Running Configuration: $CONFIG"
    echo "----------------------------------------------"

    # Initialize total search time for this configuration
    total_time=0

    # Variable to store static output from the first run of this configuration
    static_output=""

    # Run the executable NUM_RUNS times for the current configuration
    for ((i=1; i<=NUM_RUNS; i++)); do
        echo "Run #$i"

        # Run the executable with the current configuration's arguments
        $EXECUTABLE $CONFIG > "$temp_file"

        # Check if the executable ran successfully
        if [ $? -ne 0 ]; then
            echo "Error: Executable failed on run #$i for configuration: $CONFIG"
            rm "$temp_file"
            exit 1
        fi

        # On the first run, capture the static output (excluding "Search Time:")
        if [ $i -eq 1 ]; then
            # Extract all lines except those containing "Search Time:"
            static_output=$(grep -v "Search Time:" "$temp_file")
        fi

        # Extract the Search Time from the current run
        # Assumes the format "Search Time: 1306 ms"
        search_time=$(grep "Search Time:" "$temp_file" | awk '{print $3}' | tr -d 'ms')

        # Validate that search_time is a number
        if ! [[ "$search_time" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "Error: Could not parse Search Time on run #$i for configuration: $CONFIG"
            rm "$temp_file"
            exit 1
        fi

        # Add the search time to the total
        total_time=$(echo "$total_time + $search_time" | bc)

        # Optionally, print the search time of the current run
        echo "Search Time: $search_time ms"
    done

    # Compute the average search time for this configuration
    avg_time=$(echo "scale=2; $total_time / $NUM_RUNS" | bc)

    # Print the Final Output
    echo ""
    echo "Final Output:"
    echo "$static_output"
    echo "Average Search Time: $avg_time ms"
    echo ""
done

# Remove the temporary file
rm "$temp_file"