#!/bin/bash
#SBATCH --job-name=estimate
#SBATCH --output=estimate_%j.out
#SBATCH --error=estimate_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -A TEACH-IAMCSF  # Use your project ID

# Print information
echo "Running on host: $(hostname)"
echo "Start time: $(date)"

# Load modules and activate environment
source /usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh
conda activate coursework

# Check if files exist
echo "Checking for required files..."
if [ -f "estimate_parameters.py" ]; then
    echo "Python script found"
else
    echo "ERROR: Python script not found"
    exit 1
fi

if [ -f "representational.mat" ]; then
    echo "MAT file found"
else
    echo "ERROR: MAT file not found"
    exit 1
fi

# Install required packages
pip install scipy matplotlib

# Run the script
echo "Starting analysis..."
python estimate_parameters.py

echo "Job finished at: $(date)"
