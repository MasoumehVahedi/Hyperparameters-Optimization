#!/bin/bash

# Get the current directory of this script
script_dir=$(pwd)

# Define the path to your Python scripts
PREPARE_SCRIPT="prepare_data.py"
SCRIPT_PATH="main.py"

# Run the prepare script once to generate the CSV and datasets
python $PREPARE_SCRIPT

# Define directories 
QUERIES_DIR="$script_dir/Workloads"

# Define the query files
query_suffix=("0.001%" "1%" "3%")

# Iterate over each query suffix to find the matching query files
for suffix in "${query_suffix[@]}"; do      # [@] means all elements in the array
  query_file="query_ranges_${suffix}.npy"
  query_path="${QUERIES_DIR}/${query_file}"


  # Run the Python script with the required arguments
  python $SCRIPT_PATH "$query_path" "$suffix"
done  


