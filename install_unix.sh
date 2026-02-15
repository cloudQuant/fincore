#!/bin/bash

# Define variables
BUILD_DIR="build"
EGG_INFO_DIR="empyrical.egg-info"
BENCHMARKS_DIR=".benchmarks"



# Switch to the parent directory
cd ..

# Install dependencies from requirements.txt
pip install -U -r ./empyrical/requirements.txt

# Install empyrical
pip install -U --no-build-isolation ./empyrical

# Run empyrical tests with 4 parallel workers
pytest ./empyrical/tests -n 4

# Switch back to the empyrical directory
cd ./empyrical



# Remove intermediate build artifacts and egg-info
echo "Deleting intermediate files..."
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
    echo "Deleted $BUILD_DIR directory."
fi

if [ -d "$EGG_INFO_DIR" ]; then
    rm -rf "$EGG_INFO_DIR"
    echo "Deleted $EGG_INFO_DIR directory."
fi

# Remove the .benchmarks directory
if [ -d "$BENCHMARKS_DIR" ]; then
    rm -rf "$BENCHMARKS_DIR"
    echo "Deleted $BENCHMARKS_DIR directory."
fi

# Remove all .log files
echo "Deleting all .log files..."
find . -type f -name "*.log" -exec rm -f {} \;
echo "All .log files deleted."


