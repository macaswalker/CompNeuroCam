#!/bin/bash
#SBATCH --job-name=representational_analysis
#SBATCH --output=representational_%j.out
#SBATCH --error=representational_%j.err
#SBATCH --time=12:00:00  # Reduced from 24 to 12 hours
#SBATCH --mem=32G  # Reduced memory
#SBATCH --cpus-per-task=4  # Reduced CPUs
#SBATCH -A han-sl3-cpu  # CPU account from sacctmgr
#SBATCH --partition=cclake  # Confirmed partition from sinfo

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"

# Load Conda
source /usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh
conda activate coursework

# Change to working directory
cd ~/rds/hpc-work/coursework

# Check required files
if [ ! -f "representational_analysis.py" ]; then
    echo "ERROR: Python script not found"
    exit 1
fi

if [ ! -f "representational.mat" ]; then
    echo "ERROR: MAT file not found"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p representational_output

# Run the script
echo "Starting analysis..."
python representational_analysis.py

# Check script exit status
EXIT_STATUS=$?

echo "Job finished at: $(date)"
echo "Exit status: $EXIT_STATUS"

exit $EXIT_STATUS 
