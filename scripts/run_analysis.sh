#!/bin/bash
# Submit the CPU analysis chain as 6 Slurm jobs (at most 4 running at once):
#   1. labels  ->  2-4. one job per model (4 probe runs + extras)  +  5. surrogate (6 calibrated fits)
#   ->  6. overlap/transfer tables + main_table.md + figures
# Run from the repo root after cache/<model>/<dataset>/ holds the hidden states (chunk_*.npz/jsonl
# from src/extract.py) and judged.parquet.
set -euo pipefail
# Optional: AFTER=<jobid[:jobid]> makes the first job wait for upstream jobs (used by run_all.sh).
mkdir -p logs results figures
ACT='cd "$SLURM_SUBMIT_DIR"; source "${CONDA_ACTIVATE:-$HOME/miniconda3/bin/activate}" "${CONDA_ENV:-onepass}"; export OMP_NUM_THREADS=16 PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1; set -e'
SB="sbatch --parsable --partition=normal --cpus-per-task=16 --mem=48G --output=logs/%x_%j.out"

lab=$($SB --time=00:30:00 ${AFTER:+--dependency=afterok:$AFTER} --job-name=xl_labels --wrap "$ACT; python src/labels.py")
jobs=()
for m in llama qwen gemma; do
  jobs+=($($SB --time=12:00:00 --dependency=afterok:$lab --job-name=xl_model_$m --wrap "$ACT
    for t in dep_strict dep_claimant; do python src/probes.py --model $m --dataset borderlines --target \$t --heldout; done
    for ds in mkqa pooled; do python src/probes.py --model $m --dataset \$ds --target dep_strict --layer_stride 4; done
    python src/extras.py --model $m"))
done
jobs+=($($SB --time=12:00:00 --dependency=afterok:$lab --job-name=xl_surrogate --wrap "$ACT
  for P in llama qwen gemma; do for T in llama qwen gemma; do [ \$P = \$T ] || python src/surrogate.py calibrated --probe \$P --target_model \$T; done; done"))
dep=$(IFS=:; echo "${jobs[*]}")
fin=$($SB --time=00:30:00 --cpus-per-task=4 --mem=16G --dependency=afterok:$dep --job-name=xl_tables --wrap "$ACT
  for t in dep_strict dep_claimant; do python src/surrogate.py overlap --target \$t; done
  python src/surrogate.py table; python src/make_tables.py; python src/make_main_figure.py")
echo "labels=$lab model/surrogate=${jobs[*]} tables=$fin"
