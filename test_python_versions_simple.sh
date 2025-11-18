#!/bin/bash
# Simple test script for empyrical across Python versions

echo "========================================"
echo "Empyrical Python Compatibility Test"
echo "========================================"
echo

# Create results directory
mkdir -p test_results

# Create summary file
summary="test_results/summary.txt"
echo "Empyrical Test Summary" > "$summary"
echo "Tested on: $(date)" >> "$summary"
echo >> "$summary"

# Test each Python version (py38 to py313)
for v in py38 py39 py310 py311 py312 py313; do
    echo
    echo "Testing $v..."
    echo "----------------------------------------"
    
    # Check if conda environment exists
    if ! conda env list | grep -q "^$v "; then
        echo "$v: NOT FOUND - Conda environment missing" >> "$summary"
        echo "[SKIP] $v environment not found"
        continue
    fi
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$v"
    
    if [ $? -ne 0 ]; then
        echo "$v: NOT FOUND - Failed to activate environment" >> "$summary"
        echo "[SKIP] Failed to activate $v"
        continue
    fi
    
    # Get Python version
    pyver=$(python --version 2>&1)
    echo "Using $pyver"
    
    # Install and test
    echo "Installing dependencies..."
    pip install -U -r requirements.txt > "test_results/${v}_install.log" 2>&1
    
    echo "Installing empyrical..."
    pip install -U . >> "test_results/${v}_install.log" 2>&1
    
    echo "Running tests..."
    pytest tests -n 4 --tb=short > "test_results/${v}_tests.log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "$v: FAILED - $pyver" >> "$summary"
        echo "[FAIL] Tests failed for $v"
        
        # Extract failure summary
        grep -E "(FAILED|ERROR)" "test_results/${v}_tests.log" | grep -v ".py" >> "$summary" 2>/dev/null
    else
        echo "$v: PASSED - $pyver" >> "$summary"
        echo "[PASS] All tests passed for $v"
        
        # Extract success summary
        grep "passed" "test_results/${v}_tests.log" | grep "==" >> "$summary" 2>/dev/null
    fi
    
    echo >> "$summary"
    conda deactivate
done

echo
echo "========================================"
echo "Test Summary:"
echo "========================================"
cat "$summary"
echo
echo "Detailed logs: test_results/"