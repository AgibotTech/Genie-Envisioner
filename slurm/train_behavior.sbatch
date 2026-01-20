#!/bin/bash -l
#SBATCH --job-name=GE-fast-b1k
#SBATCH --partition=compute                
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/shared_work/physical_intelligence/sajjad/slurm-logs/%x-%j.out
#SBATCH --error=/shared_work/physical_intelligence/sajjad/slurm-logs/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --chdir=/shared_work/physical_intelligence/sajjad/projects/Genie-Envisioner

# --- 0. Pre-emptive Fixes (Silence Cluster Noise) ---
# Silence the /etc/profile.d/modules.sh error
exec 2> >(grep -v "/etc/profile.d/modules.sh" >&2)
# Fix the Triton autotune directory error
mkdir -p /home/sajjad/.triton/autotune

# Exit on error, treat unset variables as error
set -euo pipefail

echo "================================================================"
echo "JOB START: $(date)"
echo "NODE:      $(hostname)"
echo "WORKDIR:   $(pwd)"
echo "================================================================"

# --- 1. Environment Cleanup & Metadata Repair ---
echo "Cleaning environment and repairing corrupted metadata..."
unset PYTHONPATH
unset PYTHONHOME
module purge || true

# CRITICAL: This removes the folders causing "WARNING: Ignoring invalid distribution -orch"
# These 'ghost' folders often prevent Python from seeing other libraries like 'libero'
SITE_PACKAGES="/pm/conda/envs/users/sajjad/genie_envisioner/lib/python3.10/site-packages"
rm -rf "$SITE_PACKAGES"/~* || true

# --- 2. Conda Initialization & Activation ---
echo "Loading Conda module..."
module load conda/new || { echo "ERROR: Failed to load conda module" >&2; exit 1; }

# Sourcing the profile is mandatory for 'conda activate' to work inside SLURM scripts
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "Activating environment: genie_envisioner..."
conda activate genie_envisioner || { echo "ERROR: Failed to activate genie_envisioner" >&2; exit 1; }

# # Manual override to ensure the environment is prioritized
# export PYTHONPATH="$SITE_PACKAGES:$PYTHONPATH"

# --- 3. Sanity Check: Python & Paths ---
echo "Verifying Python path..."
PYTHON_PATH=$(which python)
echo "Python location: $PYTHON_PATH"

if [[ "$PYTHON_PATH" == "/usr/bin/"* ]]; then
    echo "ERROR: System Python detected! Conda environment did not override PATH." >&2
    exit 1
fi

# --- 4. Sanity Check: Library Integrity (The 'Libero' Check) ---
echo "Testing Python library imports..."
# Check for 'libero' specifically since that was your main failure point
python -c "
import sys
import torch
import libero
print(f'SUCCESS: Python {sys.version}')
print(f'SUCCESS: Torch {torch.__version__} loaded')
print(f'SUCCESS: Libero loaded from {libero.__file__}')
" || { 
    echo "ERROR: Libero or Torch still not found! Attempting emergency forced reinstall..." >&2
    python -m pip install --no-cache-dir libero
}

# --- 5. Sanity Check: GPU Visibility ---
echo "Checking GPU availability..."
python -c "import torch; dev_count = torch.cuda.device_count(); print(f'Torch sees {dev_count} GPUs'); exit(0 if dev_count >= 1 else 1)" || { 
    echo "ERROR: GPU mismatch. Expected at least 1, saw $(python -c 'import torch; print(torch.cuda.device_count())')" >&2
    exit 1 
}

# --- 6. Execution ---
echo "================================================================"
echo "STARTING WORKLOAD: scripts/train.sh main.py"
echo "================================================================"

# Use 'export -f' to ensure conda functions carry over to the sub-shell if needed
export -f conda 2>/dev/null || true

# This finds a random unused port between 10000 and 65000
export MASTER_PORT=$(shuf -i 20000-60000 -n 1)
echo "Using Master Port: $MASTER_PORT"

# adapt video model
bash scripts/train.sh main.py configs/ltx_model/behavior/video_model_behavior.yaml
# fine-tune action model
# bash scripts/train.sh main.py configs/ltx_model/behavior/action_model_behavior.yaml
# bash scripts/train.sh main.py configs/ltx_model/behavior/action_model_wa_behavior.yaml

echo "================================================================"
echo "JOB FINISHED: $(date)"
echo "================================================================"