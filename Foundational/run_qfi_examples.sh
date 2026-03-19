#!/bin/bash
# Example scripts for running QFI analysis

# Example 1: Basic usage with default parameters
echo "Example 1: Basic QFI analysis"
python plot_QFI_disorder.py logs/run_2026-02-13_22-44-00/

# Example 2: With more test configurations for better statistics
echo ""
echo "Example 2: QFI analysis with 10 test configs per h0"
python plot_QFI_disorder.py logs/run_2026-02-13_22-44-00/ --n-test 10

# Example 3: Custom output location
echo ""
echo "Example 3: QFI analysis with custom output"
python plot_QFI_disorder.py logs/run_2026-02-13_22-44-00/ --output custom_qfi_plot.png

# Example 4: Batch processing multiple runs (if you have several)
echo ""
echo "Example 4: Processing all runs in logs/"
for run_dir in logs/run_*; do
    if [ -d "$run_dir" ]; then
        echo "Processing $run_dir..."
        python plot_QFI_disorder.py "$run_dir" --n-test 5
    fi
done
