# One Pass, Many Languages: Predicting Cross-Lingual Answer Change from a Single Hidden State

> This work was done during the **[Apart Research](https://apartresearch.com) AI x Epistemics research sprint** (13–16 Nov 2026). It is hackathon research: the results are preliminary, and the caveats are listed [below](#limitations-and-caveats).

## Motivation

People increasingly ask LLMs factual questions in their own language, and they assume the answer doesn't depend on the language they asked in. It often does. Ask a model which country a disputed territory belongs to, or an ordinary factual question, in English and then in Arabic, Russian or Chinese, and you can get different answers. A user who asks in only one language never sees the disagreement. So the model's "knowledge" can quietly differ by language community, which is an epistemic problem.

Checking for this directly means translating the question, asking the model in many languages, and comparing the answers. That is expensive, it needs a judge to compare free-text answers, and you have to guess which languages to check.

## Idea

Maybe the model already "knows" when its answer is unstable across languages. We test whether this is readable from **a single forward pass in the user's own language**:

1. Ask the question once, in the user's language.
2. Read the hidden state at the last prompt token, before any answer is generated.
3. A linear probe predicts whether the *same model* would give a different answer if asked in another language.

Running the probe needs no translation, no extra generations and no judge. Translations and the judge are used only to build the training labels.

We compare the probe against single-pass uncertainty baselines (first-token entropy, mean log-probability and similar) on two datasets:
- **BorderLines**: 242 disputed territories, each asked in the languages of its claimant countries plus English.
- **MKQA**: about 2,000 general factual questions, each asked in 5 languages.

We test three open models: Llama-3.1-8B, Qwen2.5-7B and Gemma-3-12B.

## Preliminary results

AUROC [95% CI, bootstrap over questions]. The primary label is "the answer changes across languages". Evaluation is 5-fold cross-validation grouped by question.

| | Llama-3.1-8B | Qwen2.5-7B | Gemma-3-12B |
|---|---|---|---|
| **BorderLines** (242 disputed territories): probe | **0.743** [0.684, 0.801] | **0.710** [0.639, 0.778] | **0.769** [0.713, 0.823] |
| Best single-pass baseline | 0.632 (first-token entropy) | 0.656 (−mean log-prob) | 0.692 (−mean log-prob) |
| Probe − best baseline (one-sided p) | +0.111 (0.0005) | +0.054 (0.03)† | +0.076 (0.009) |
| **MKQA** (≈2,000 questions × 5 languages): probe | **0.754** [0.722, 0.783] | **0.768** [0.737, 0.796] | **0.750** [0.726, 0.773] |
| Probe − best baseline | +0.126 (<0.0005) | +0.125 (<0.0005) | +0.105 (<0.0005) |
| Train MKQA → test BorderLines | 0.669 [0.604, 0.729] | 0.701 [0.627, 0.773] | 0.804 [0.753, 0.852] |

† The 95% CI of the difference includes 0. The probe does **not** predict the narrower label "names a different claimant in different languages" (AUROC 0.43–0.51). See the report.

## Repository layout

```
src/
  translate_templates.py  translate "Which country does XX belong to?" + back-translate (Llama-3.3-70B)   [GPU x2]
  build_prompts.py        BorderLines open-ended prompts (native territory names)                         [CPU]
  build_mkqa.py           MKQA sample: 2,000 questions x {en, es, ar, zh_cn, ru}                         [CPU]
  extract.py              greedy answer + last-prompt-token hidden state at every layer                  [GPU x1]
  judge.py / judge_mkqa.py  Llama-3.3-70B judge: canonical answer labels / answer groups                 [GPU x2]
  labels.py               question-level labels (dep_strict, dep_claimant, spread)                       [CPU]
  probes.py               linear probe + baselines, grouped CV, bootstrap CIs, layer sweep, held-out lang [CPU]
  extras.py               cross-dataset transfer + MLP probe                                             [CPU]
  surrogate.py            cross-model flip overlap + probe transfer (exploratory)                        [CPU]
  make_tables.py          results/main_table.md + per-dataset layer-sweep figures                        [CPU]
  make_main_figure.py     figures/layer_sweep_main.pdf (paper Fig. 2)                                    [CPU]
slurm/                    sbatch wrappers (submit from the repo root)
scripts/run_all.sh        end-to-end: extract (3 GPU jobs) -> judge (optional rerun) -> analysis
scripts/run_analysis.sh   CPU analysis chain only (6 Slurm jobs), from existing hidden states
scripts/compare_results.py  diff regenerated results/ against a reference copy
data/                     translated templates + final prompt tables
cache/<model>/<dataset>/  released model answers + judge labels (judged.parquet, labels.parquet, meta.json)
results/                  all numbers in the report (JSON) + main_table.md
figures/                  layer-sweep figures
judge_audit.csv           50-item sheet for a human audit of judge labels (not yet filled in)
```

**Not included:** hidden states (about 14 GB) and out-of-fold predictions. Regenerate them as described below.

## Setup

```bash
conda env create -f environment.yml && conda activate onepass
# or, in an existing Python 3.10 env: pip install -r requirements.txt
```

**Models** (Hugging Face; the gated ones need access):
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `google/gemma-3-12b-it`
- `meta-llama/Llama-3.3-70B-Instruct` (judge and translator, bf16, 2×80 GB)
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (text baseline)

**Data:**
```bash
git clone https://github.com/manestay/borderlines.git data/raw/borderlines && git -C data/raw/borderlines checkout 5de2146
git clone https://github.com/apple/ml-mkqa.git data/raw/ml-mkqa && git -C data/raw/ml-mkqa checkout f964fe1
```
BorderLines territories and queries are also loaded from the HF hub (`manestay/borderlines`).

**Slurm wrappers:** they read `CONDA_ACTIVATE` (path to conda's `activate`) and `CONDA_ENV` (environment name, default `onepass`), and set `HF_HUB_OFFLINE=1`. Export `HF_HOME` if your model cache is not in the default location.

## Reproduce

### A. Tables and figures from the released numbers (seconds)
```bash
python src/make_tables.py        # -> results/main_table.md, figures/layer_sweep_{borderlines,mkqa,pooled}.pdf
python src/make_main_figure.py   # -> figures/layer_sweep_main.pdf
```

### B. Full pipeline
**One command** (Slurm; from the repo root):
```bash
./scripts/run_all.sh            # extract hidden states -> analysis, reusing the released judge outputs
./scripts/run_all.sh --rejudge  # also rerun the Llama-3.3-70B judge
```
The table below lists the individual steps.

The steps below are run from the repo root. Steps 1–2 are already done: their outputs are in `data/`.

| Step | Command | Output |
|---|---|---|
| 1. Templates | `sbatch slurm/translate_templates.sbatch` | `data/templates_open.tsv` (`templates_open_manual.tsv` = 4 hand-written fixes) |
| 2. Prompts | `python src/build_prompts.py --drop ht,lo,mn,my,so,mg,km,sn,sq` ; `python src/build_mkqa.py` | `data/prompts_{borderlines,mkqa}.parquet` |
| 3. Extract | `sbatch slurm/extract.sbatch --model llama --prompts data/prompts_borderlines.parquet --out cache/llama/borderlines` (repeat for `qwen`, `gemma`; MKQA: `--prompts data/prompts_mkqa.parquet --out cache/<m>/mkqa --max_new_tokens 64`; Gemma: add `--save_dtype float32`) | `cache/<m>/<ds>/chunk_*.{npz,jsonl}` |
| 4. Judge | `sbatch slurm/judge.sbatch` ; `sbatch slurm/judge_mkqa.sbatch` (skipped if `judged.parquet` exists, i.e. the released ones) | `cache/<m>/<ds>/judged.parquet` |
| 5. Analysis | `./scripts/run_analysis.sh` (6 Slurm jobs: labels → per-model probes + extras, surrogate → tables/figures) | `results/`, `figures/` |

**Notes on the pipeline:**
- Step 5 needs the hidden states from step 3. The probes read `cache/<m>/<ds>/chunk_*.npz`.
- Extraction is resumable, and greedy decoding makes it deterministic.
- Gemma's hidden states must be saved in float32, because one activation channel overflows float16.

**Checking the numbers.** Copy the released `results/` to `expected/results/` before step 5, then run:

```bash
python scripts/compare_results.py expected/results
```

### Where each paper item comes from

| Paper item | File |
|---|---|
| Table 1 (main results) | `results/probes_<m>_{borderlines,mkqa}_dep_strict.json`, `results/extras_<m>.json` (cross-dataset) |
| Fig. 2 (layer sweep) | `figures/layer_sweep_main.pdf` ← `results/layer_sweep_<m>_{borderlines,mkqa}_dep_strict.json` |
| Full baseline tables, held-out language, AUPRC, layers chosen | `results/main_table.md` |
| Secondary label (claimant change) | `results/probes_<m>_borderlines_dep_claimant.json`, `figures/layer_sweep_borderlines.pdf` |
| MLP vs linear, cross-dataset | `results/extras_<m>.json` |
| Pooled probe | `results/probes_<m>_pooled_dep_strict.json` |
| Cross-model surrogate | `results/flip_overlap_borderlines_*.json`, `results/transfer_borderlines_dep_strict.json` |
| Class balance, answer mix | `results/class_balance.json`, `results/class_balance_mkqa.json` |

## Method in brief
- **Rows:** one row per (question, language) pair. The label is question-level: does the model's canonicalised answer differ across the languages tested?
- **Probe:** StandardScaler → PCA(64) → L2 logistic regression on one layer's last-prompt-token state.
- **Model selection:** the layer and C ∈ {0.001, 0.01, 0.1} are chosen by an inner 3-fold grouped CV inside each training fold. The outer loop is 5-fold GroupKFold by question, so all languages of a question share a fold.
- **Statistics:** CIs come from 2,000 bootstrap resamples over questions, and the probe is compared with each baseline by a paired bootstrap test.

## Limitations and caveats
- **LLM-made labels and translations.** Labels come from an LLM judge, and prompt templates were LLM-translated and checked only by back-translation. Human verification is pending (`judge_audit.csv`). Four templates (bs, sr, sl, fa) were written by hand and not checked by native speakers.
- **Small sample.** There are 242 BorderLines territories, so CIs are wide.
- **Scope of "stable".** It means stable across the languages tested, under greedy decoding.
- **Access.** The method is white-box: it needs the model's hidden states.
- **Dual use.** The probe flags instability but does not reveal alternative answers.

## Citation
```bibtex
@misc{onepass2026,
  title  = {One Pass, Many Languages: Predicting Cross-Lingual Answer Change from a Single Hidden State},
  author = {Kala, Kalash},
  year   = {2026},
  note   = {Apart Research AI x Epistemics Research Sprint},
  url    = {https://github.com/kalash-kala/one-pass-many-languages}
}
```

## License
MIT. BorderLines and MKQA are subject to their own licences.
