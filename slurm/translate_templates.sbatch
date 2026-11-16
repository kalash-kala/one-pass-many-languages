#!/bin/bash
#SBATCH --job-name=xl_translate
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the repo root
source "${CONDA_ACTIVATE:-$HOME/miniconda3/bin/activate}" "${CONDA_ENV:-onepass}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python src/translate_templates.py "$@"
