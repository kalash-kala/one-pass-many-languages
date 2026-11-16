#!/bin/bash
# End-to-end reproduction on Slurm, from the prompt tables in data/ to results/ and figures/.
#   1. extract (GPU x1): one job per model, BorderLines then MKQA        -> 3 GPU jobs at once
#   2. judge   (GPU x2): BorderLines judge then MKQA judge, one job      -> after all extracts
#   3. analysis (CPU):   scripts/run_analysis.sh (6 jobs)               -> after the judge
# Usage (from the repo root):
#   ./scripts/run_all.sh            # reuse the released judge outputs (cache/*/*/judged.parquet)
#   ./scripts/run_all.sh --rejudge  # delete them and rerun the 70B judge too
# Templates/prompts (src/translate_templates.py, build_prompts.py, build_mkqa.py) are already in data/;
# see README step 1-2 to regenerate them.
set -euo pipefail
mkdir -p logs
if [ "${1:-}" = "--rejudge" ]; then rm -f cache/*/borderlines/judged.parquet cache/*/mkqa/judged.parquet; fi

ext=()
for m in llama qwen gemma; do
  dtype=float16; [ $m = gemma ] && dtype=float32      # Gemma overflows float16 (one activation channel)
  ext+=($(sbatch --parsable --job-name=xl_ext_$m --time=12:00:00 --wrap "cd \"\$SLURM_SUBMIT_DIR\"
    set -e
    bash slurm/extract.sbatch --model $m --prompts data/prompts_borderlines.parquet --out cache/$m/borderlines --save_dtype $dtype
    bash slurm/extract.sbatch --model $m --prompts data/prompts_mkqa.parquet --out cache/$m/mkqa --save_dtype $dtype --max_new_tokens 64" \
    --partition=normal --gres=gpu:1 --cpus-per-task=8 --mem=96G --output=logs/%x_%j.out))
done
dep=$(IFS=:; echo "${ext[*]}")
judge=$(sbatch --parsable --dependency=afterok:$dep --job-name=xl_judge_all --time=12:00:00 --wrap "cd \"\$SLURM_SUBMIT_DIR\"
  set -e; bash slurm/judge.sbatch; bash slurm/judge_mkqa.sbatch" \
  --partition=normal --gres=gpu:2 --cpus-per-task=8 --mem=200G --output=logs/%x_%j.out)
echo "extract=${ext[*]} judge=$judge"
AFTER=$judge ./scripts/run_analysis.sh
