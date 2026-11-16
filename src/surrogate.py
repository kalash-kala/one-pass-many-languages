"""Surrogate analysis (exploratory).

  overlap   : per model pair, Jaccard of language-dependent question sets + Cohen's kappa (the
              ceiling on any surrogate).  -> results/flip_overlap_<dataset>_<target>.json
  calibrated: probe trained on model P's hidden states with model T's labels (as after a one-off
              audit of T), same grouped CV + inner selection as probes.py.
              -> results/oof_surrogate_<P>_on_<T>_<dataset>_<target>.parquet
  table     : 3x3 tables. Zero-shot = P's own probe (oof from probes.py, trained on P's labels)
              scored against T's labels. Calibrated = above. Diagonal = white-box result.
              -> results/transfer_<dataset>_<target>.json
"""
import argparse, itertools, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold

from probes import fit_hidden, load, score

MODELS = ["llama", "qwen", "gemma"]


def labels_by_group(model, dataset, target):
    sub = dataset
    d = pd.read_parquet(f"cache/{model}/{sub}/labels.parquet")
    key = "territory_idx" if sub == "borderlines" else "group_id"
    return d.dropna(subset=[target]).groupby(key)[target].first().astype(int)


def overlap(dataset, target):
    L = {m: labels_by_group(m, dataset, target) for m in MODELS}
    out = {}
    for a, b in itertools.combinations(MODELS, 2):
        idx = L[a].index.intersection(L[b].index)
        x, y = L[a].loc[idx].to_numpy(), L[b].loc[idx].to_numpy()
        inter, union = ((x == 1) & (y == 1)).sum(), ((x == 1) | (y == 1)).sum()
        out[f"{a}-{b}"] = dict(n=int(len(idx)), jaccard=round(float(inter / union), 4) if union else None,
                               kappa=round(float(cohen_kappa_score(x, y)), 4),
                               agreement=round(float((x == y).mean()), 4),
                               rate_a=round(float(x.mean()), 4), rate_b=round(float(y.mean()), 4))
    Path(f"results/flip_overlap_{dataset}_{target}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


def calibrated(P, T, dataset, target, stride=2):
    dP, H = load(P, dataset, target)
    dT, _ = load(T, dataset, target)
    yT = dT.set_index("prompt_id")[target].astype(int)
    keep = dP.prompt_id.isin(yT.index).to_numpy()
    dP, H = dP[keep].reset_index(drop=True), H[keep]
    y = yT.loc[dP.prompt_id].to_numpy(); groups = dP.group.to_numpy(); langs = dP.lang.to_numpy()
    folds = list(GroupKFold(5).split(H, y, groups))
    oof, ch = fit_hidden(H, y, groups, langs, folds, list(range(0, H.shape[1], stride)), [0.001, 0.01, 0.1])
    o = dP[["prompt_id", "group"]].copy(); o["y"] = y; o["score"] = oof
    o.to_parquet(f"results/oof_surrogate_{P}_on_{T}_{dataset}_{target}.parquet", index=False)
    print(P, "on", T, score(y, oof, groups)["auroc"], ch)


def table(dataset, target):
    O = {m: pd.read_parquet(f"results/oof_{m}_{dataset}_{target}.parquet").set_index("prompt_id") for m in MODELS}
    out = {"zero_shot": {}, "calibrated": {}}
    for P in MODELS:
        out["zero_shot"][P], out["calibrated"][P] = {}, {}
        for T in MODELS:
            idx = O[P].index.intersection(O[T].index)
            y, s, g = O[T].loc[idx, "y"].to_numpy(), O[P].loc[idx, "A_linear_probe"].to_numpy(), O[T].loc[idx, "group"].to_numpy()
            out["zero_shot"][P][T] = score(y, s, g)
            if P == T:
                out["calibrated"][P][T] = out["zero_shot"][P][T]
            else:
                f = Path(f"results/oof_surrogate_{P}_on_{T}_{dataset}_{target}.parquet")
                if f.exists():
                    c = pd.read_parquet(f)
                    out["calibrated"][P][T] = score(c.y.to_numpy(), c.score.to_numpy(), c.group.to_numpy())
    Path(f"results/transfer_{dataset}_{target}.json").write_text(json.dumps(out, indent=2))
    for k in out:
        print(f"\n{k} (rows = probed model, cols = target model), AUROC:")
        print(pd.DataFrame({T: {P: out[k][P].get(T, {}).get("auroc") for P in MODELS} for T in MODELS}).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["overlap", "calibrated", "table"])
    ap.add_argument("--probe"); ap.add_argument("--target_model")
    ap.add_argument("--dataset", default="borderlines")
    ap.add_argument("--target", default="dep_strict")
    a = ap.parse_args()
    if a.mode == "overlap":
        overlap(a.dataset, a.target)
    elif a.mode == "calibrated":
        calibrated(a.probe, a.target_model, a.dataset, a.target)
    else:
        table(a.dataset, a.target)


if __name__ == "__main__":
    main()
