#!/bin/bash
# Usage: sbatch --job-name=xl_probe_X slurm/cpu.sbatch src/probes.py --model llama
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the repo root
source "${CONDA_ACTIVATE:-$HOME/miniconda3/bin/activate}" "${CONDA_ENV:-onepass}"
export OMP_NUM_THREADS=16 PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1
python "$@"
