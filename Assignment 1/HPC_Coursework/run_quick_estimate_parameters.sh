#!/bin/bash
#SBATCH --job-name=quick_estimate_parameters
#SBATCH --output=quick_estimate_parameters_%j.out
#SBATCH --error=quick_estimate_parameters_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -A han-sl3-cpu
#SBATCH --partition=cclake

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"

# Load Conda
source /usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh
conda activate coursework

# Change to working directory
cd ~/rds/hpc-work/coursework

# Check if script exists
if [ ! -f "quick_estimate_parameters.py" ]; then
    echo "ERROR: quick_estimate_parameters.py not found"
    exit 1
fi

# Run the script
echo "Starting quick parameter estimation..."
python quick_estimate_parameters.py

# Check script exit status
EXIT_STATUS=$?

echo "Job finished at: $(date)"
echo "Exit status: $EXIT_STATUS"

exit $EXIT_STATUS
