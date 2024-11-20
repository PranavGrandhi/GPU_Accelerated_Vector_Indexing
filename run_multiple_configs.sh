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

# ----------------------------
# End of Configuration Section
# ----------------------------

# Function to display usage
usage() {
    echo "Usage: $0 <experiment_config_file> [<num_runs>]"
    echo " - <experiment_config_file>: Path to the configuration file containing parameters for the experiment."
    echo " - <num_runs>: (Optional) Number of runs per configuration. Defaults to $DEFAULT_NUM_RUNS."
    echo "Example:"
    echo "   $0 experiment1_config.txt 10"
    echo "   Runs configurations in 'experiment1_config.txt' 10 times."
}

# Parse arguments
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    usage
    exit 1
fi

CONFIG_FILE=$1
NUM_RUNS=${2:-$DEFAULT_NUM_RUNS}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found."
    exit 1
fi

if [[ ! "$NUM_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: <num_runs> must be a positive integer."
    usage
    exit 1
fi

# Check if the executable exists and is executable
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: Executable '$EXECUTABLE' not found or not executable."
    exit 1
fi

# Temporary file to store the output of each run
temp_file=$(mktemp)

# Iterate over each parameter configuration in the configuration file
while IFS= read -r CONFIG; do
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
done < "$CONFIG_FILE"

# Remove the temporary file
rm "$temp_file"
