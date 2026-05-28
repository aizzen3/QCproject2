#!/bin/bash
#SBATCH -p fat                # high-RAM nodes
#SBATCH --job-name=qc_run
#SBATCH --output=qc_%j.log    # log file (%j = job id)
#SBATCH --time=1-00:00:00       # max runtime
#SBATCH --mem=512G            # request RAM
#SBATCH --cpus-per-task=8     # number of CPU cores

set -euo pipefail

cd ~/QCproject2
source .venv/bin/activate

# Optional: make multi-thread libs respect the CPUs you asked for
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK  RAM: $SLURM_MEM_PER_NODE"

python smooth_sq.py
