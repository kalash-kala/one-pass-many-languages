"""Extras: cross-dataset transfer and D. small MLP probe (primary label dep_strict).

cross : train probe A on one dataset (layer + C chosen by inner grouped 3-fold CV on the training
        dataset only), test on the other dataset. Both directions.
mlp   : one hidden layer (64 units, early stopping on a 10% validation split), on PCA-64 of the
        hidden state at the modal layer chosen by probe A for that model/dataset; same outer
        grouped 5-fold CV. Compared with probe A on the same rows.
Writes results/extras_<model>.json
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probes import load, lr_probe, paired_diff, score, select


def cross(model, src, dst, stride):
    ds, Hs = load(model, src, "dep_strict")
    dd, Hd = load(model, dst, "dep_strict")
    ys, gs, ls = ds.dep_strict.astype(int).to_numpy(), ds.group.to_numpy(), ds.lang.to_numpy()
    yd, gd = dd.dep_strict.astype(int).to_numpy(), dd.group.to_numpy()
    s, L, C = select(Hs, ys, gs, ls, np.arange(len(ys)), list(range(0, Hs.shape[1], stride)),
                     [0.001, 0.01, 0.1], center=False)
    p = lr_probe(C, min(64, len(ys) - 1)).fit(Hs[:, L], ys)
    return {**score(yd, p.predict_proba(Hd[:, L])[:, 1], gd), "layer": int(L), "C": C,
            "inner_auroc_on_train": round(s, 4)}


def mlp(model, dataset):
    r = json.loads(Path(f"results/probes_{model}_{dataset}_dep_strict.json").read_text())
    L = int(pd.Series([c["layer"] for c in r["A_linear_probe"]["chosen"]]).mode()[0])
    d, H = load(model, dataset, "dep_strict")
    y, g = d.dep_strict.astype(int).to_numpy(), d.group.to_numpy()
    oofA = pd.read_parquet(f"results/oof_{model}_{dataset}_dep_strict.parquet").set_index("prompt_id")
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(H, y, g):
        m = make_pipeline(StandardScaler(), PCA(min(64, len(tr) - 1), random_state=0),
                          MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, early_stopping=True,
                                        validation_fraction=0.1, max_iter=500, random_state=0))
        oof[te] = m.fit(H[tr, L], y[tr]).predict_proba(H[te, L])[:, 1]
    a = oofA.loc[d.prompt_id, "A_linear_probe"].to_numpy()
    return {**score(y, oof, g), "layer": L, "mlp_minus_linear": paired_diff(y, oof, a, g)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    res = {"cross_mkqa_to_borderlines": cross(a.model, "mkqa", "borderlines", 4),
           "cross_borderlines_to_mkqa": cross(a.model, "borderlines", "mkqa", 2),
           "mlp_borderlines": mlp(a.model, "borderlines"),
           "mlp_mkqa": mlp(a.model, "mkqa")}
    Path(f"results/extras_{a.model}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
