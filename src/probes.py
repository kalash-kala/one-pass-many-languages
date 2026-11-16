"""White-box probes + baselines with grouped CV. Saves per-row out-of-fold predictions.

Rows = (question, language). Target = question-level label (dep_strict / dep_claimant).
Datasets: borderlines (group = territory), mkqa (group = MKQA question), pooled (both).
Protocol:
  * outer 5-fold GroupKFold by group; probe layer and C chosen on inner 3-fold grouped CV
    inside the training fold only;
  * AUROC (primary) + AUPRC on pooled out-of-fold predictions;
  * 95% CI: bootstrap over groups (2000 resamples).
Methods:
  A  standardise -> PCA(<=64, fit on train) -> L2 logistic regression
  C  as A after subtracting per-language mean hidden state (train fold; unseen lang -> global mean)
  B  ridge onto spread -> Spearman (dep_strict only)
Baselines: first-token entropy, -mean answer logprob, answer length, metadata LR,
           question-text multilingual embedding LR.
Extra: held-out language test (--heldout): for each of the most frequent languages L, train on
       rows with lang != L in the training groups, test on rows with lang == L in held-out groups.
Outputs: results/probes_<model>_<dataset>_<target>.json, results/oof_<...>.parquet,
         results/layer_sweep_<...>.json
"""
import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG = np.random.default_rng(0)
EMB = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


# ---------------------------------------------------------------- data
def load_hidden(model, sub):
    hs, ids = [], []
    for f in sorted(glob.glob(f"cache/{model}/{sub}/chunk_*.npz")):
        z = np.load(f); hs.append(z["hidden"].astype(np.float32)); ids += z["prompt_id"].tolist()
    return np.concatenate(hs), ids


def load_answers(model, sub):
    return pd.concat([pd.read_json(f, lines=True) for f in
                      sorted(glob.glob(f"cache/{model}/{sub}/chunk_*.jsonl"))]).set_index("prompt_id")


def load(model, dataset, target):
    parts = []
    for sub in (["borderlines", "mkqa"] if dataset == "pooled" else [dataset]):
        d = pd.read_parquet(f"cache/{model}/{sub}/labels.parquet")
        if target not in d:
            continue
        d = d[d[target].notna()].copy()
        d["dataset"] = sub
        d["group"] = (d.territory_idx.astype(str).radd("bl_") if sub == "borderlines" else d.group_id)
        if sub == "mkqa":
            d["is_claimant_lang"] = False
            d["n_claimants"] = 0
        else:
            d["n_claimants"] = d.claimants.map(len)
        H, ids = load_hidden(model, sub)
        pos = {p: i for i, p in enumerate(ids)}
        A = load_answers(model, sub)
        d["mean_logprob"] = [np.mean(A.loc[p, "token_logprobs"]) if A.loc[p, "n_tokens"] else 0.0
                             for p in d.prompt_id]
        parts.append((d.reset_index(drop=True), H[[pos[p] for p in d.prompt_id]]))
    d = pd.concat([p[0] for p in parts], ignore_index=True)
    H = np.concatenate([p[1] for p in parts])
    return d, H


# ---------------------------------------------------------------- metrics
def boot_ci(y, s, groups, fn, n=2000):
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        take = np.concatenate([idx[g] for g in RNG.choice(ug, len(ug))])
        if len(np.unique(y[take])) < 2:
            continue
        vals.append(fn(y[take], s[take]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def score(y, s, groups):
    return dict(auroc=round(float(roc_auc_score(y, s)), 4), auroc_ci=boot_ci(y, s, groups, roc_auc_score),
                auprc=round(float(average_precision_score(y, s)), 4),
                auprc_ci=boot_ci(y, s, groups, average_precision_score),
                base_rate=round(float(y.mean()), 4))


def paired_diff(y, s1, s2, groups, n=2000):
    """AUROC(s1) - AUROC(s2), bootstrap over groups; returns diff, CI, one-sided p (diff<=0)."""
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    obs = roc_auc_score(y, s1) - roc_auc_score(y, s2); vals = []
    for _ in range(n):
        t = np.concatenate([idx[g] for g in RNG.choice(ug, len(ug))])
        if len(np.unique(y[t])) < 2:
            continue
        vals.append(roc_auc_score(y[t], s1[t]) - roc_auc_score(y[t], s2[t]))
    vals = np.array(vals)
    return dict(diff=round(float(obs), 4), ci=[round(float(np.percentile(vals, 2.5)), 4),
                                               round(float(np.percentile(vals, 97.5)), 4)],
                p_one_sided=round(float((vals <= 0).mean()), 4))


# ---------------------------------------------------------------- models
def lr_probe(C, n_comp):
    return make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=0),
                         LogisticRegression(C=C, max_iter=2000))


def lang_center(Xtr, ltr, Xte, lte):
    mu = {l: Xtr[ltr == l].mean(0) for l in np.unique(ltr)}
    g = Xtr.mean(0)
    return Xtr - np.stack([mu[l] for l in ltr]), Xte - np.stack([mu.get(l, g) for l in lte])


def select(H, y, groups, langs, tr, layers, Cs, center, inner=3):
    best = (-1, None, None)
    for L in layers:
        for C in Cs:
            ss = []
            for itr, iva in GroupKFold(inner).split(tr, y[tr], groups[tr]):
                a, b = tr[itr], tr[iva]
                if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
                    continue
                Xa, Xb = H[a, L], H[b, L]
                if center:
                    Xa, Xb = lang_center(Xa, langs[a], Xb, langs[b])
                p = lr_probe(C, min(64, len(a) - 1)).fit(Xa, y[a])
                ss.append(roc_auc_score(y[b], p.predict_proba(Xb)[:, 1]))
            if ss and np.mean(ss) > best[0]:
                best = (float(np.mean(ss)), L, C)
    return best


def fit_hidden(H, y, groups, langs, folds, layers, Cs, center=False):
    oof = np.zeros(len(y)); chosen = []
    for tr, te in folds:
        s, L, C = select(H, y, groups, langs, tr, layers, Cs, center)
        Xtr, Xte = H[tr, L], H[te, L]
        if center:
            Xtr, Xte = lang_center(Xtr, langs[tr], Xte, langs[te])
        oof[te] = lr_probe(C, min(64, len(tr) - 1)).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        chosen.append(dict(layer=int(L), C=C, inner_auroc=round(s, 4)))
    return oof, chosen


def fit_simple(X, y, groups, folds, Cs=(0.01, 0.1, 1.0)):
    oof = np.zeros(len(y))
    for tr, te in folds:
        best = (-1, 1.0)
        for C in Cs:
            ss = []
            for a, b in GroupKFold(3).split(tr, y[tr], groups[tr]):
                a, b = tr[a], tr[b]
                if len(set(y[a])) < 2 or len(set(y[b])) < 2:
                    continue
                p = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[a], y[a])
                ss.append(roc_auc_score(y[b], p.predict_proba(X[b])[:, 1]))
            if ss and np.mean(ss) > best[0]:
                best = (np.mean(ss), C)
        oof[te] = make_pipeline(StandardScaler(), LogisticRegression(C=best[1], max_iter=2000)
                                ).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def meta_features(d):
    X = d[["lang", "is_claimant_lang", "dataset"]].astype(str)
    enc = OneHotEncoder(handle_unknown="ignore").fit(X)  # categories only; no label information
    return np.hstack([enc.transform(X).toarray(), d[["n_claimants"]].to_numpy(float)])


def text_features(d):
    from sentence_transformers import SentenceTransformer
    pr = pd.concat([pd.read_parquet(f"data/prompts_{s}.parquet")[["prompt_id", "prompt"]]
                    for s in d.dataset.unique()]).set_index("prompt_id")
    enc = SentenceTransformer(EMB, device="cpu")
    return enc.encode(pr.loc[d.prompt_id, "prompt"].tolist(), batch_size=64, normalize_embeddings=True)


def layer_sweep(H, y, folds, C=0.01):
    out = []
    for L in range(H.shape[1]):
        oof = np.zeros(len(y))
        for tr, te in folds:
            oof[te] = lr_probe(C, min(64, len(tr) - 1)).fit(H[tr, L], y[tr]).predict_proba(H[te, L])[:, 1]
        out.append(dict(layer=L, auroc=round(float(roc_auc_score(y, oof)), 4)))
    return out


def heldout_language(H, y, groups, langs, folds, layer, C, top=8):
    res = {}
    for L in pd.Series(langs).value_counts().index[:top]:
        s, yt, gt = [], [], []
        for tr, te in folds:
            tr2 = tr[langs[tr] != L]; te2 = te[langs[te] == L]
            if len(te2) == 0 or len(np.unique(y[tr2])) < 2:
                continue
            p = lr_probe(C, min(64, len(tr2) - 1)).fit(H[tr2, layer], y[tr2])
            s.append(p.predict_proba(H[te2, layer])[:, 1]); yt.append(y[te2]); gt.append(groups[te2])
        s, yt, gt = np.concatenate(s), np.concatenate(yt), np.concatenate(gt)
        res[L] = ({**score(yt, s, gt), "n": int(len(yt))} if len(np.unique(yt)) == 2
                  else {"n": int(len(yt)), "note": "single class"})
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="borderlines", choices=["borderlines", "mkqa", "pooled"])
    ap.add_argument("--target", default="dep_strict", choices=["dep_strict", "dep_claimant"])
    ap.add_argument("--layer_stride", type=int, default=2)
    ap.add_argument("--heldout", action="store_true")
    args = ap.parse_args()
    tag = f"{args.model}_{args.dataset}_{args.target}"

    d, H = load(args.model, args.dataset, args.target)
    y = d[args.target].astype(int).to_numpy(); groups = d.group.to_numpy(); langs = d.lang.to_numpy()
    folds = list(GroupKFold(5).split(H, y, groups))
    S = {}  # out-of-fold scores
    S["baseline_first_token_entropy"] = d.first_token_entropy.to_numpy()
    S["baseline_neg_mean_logprob"] = -d.mean_logprob.to_numpy()
    S["baseline_answer_length"] = d.n_tokens.to_numpy(float)
    S["baseline_metadata"] = fit_simple(meta_features(d), y, groups, folds)
    S["baseline_question_text"] = fit_simple(text_features(d), y, groups, folds)

    layers = list(range(0, H.shape[1], args.layer_stride)); Cs = [0.001, 0.01, 0.1]
    S["A_linear_probe"], chA = fit_hidden(H, y, groups, langs, folds, layers, Cs)
    S["C_lang_normalised"], chC = fit_hidden(H, y, groups, langs, folds, layers, Cs, center=True)

    res = dict(model=args.model, dataset=args.dataset, target=args.target, n_rows=len(d),
               n_groups=int(len(np.unique(groups))))
    for k, s in S.items():
        res[k] = score(y, s, groups)
    res["A_linear_probe"]["chosen"] = chA; res["C_lang_normalised"]["chosen"] = chC
    base_keys = [k for k in S if k.startswith("baseline_")]
    best_base = max(base_keys, key=lambda k: res[k]["auroc"])
    res["paired_A_vs_best_baseline"] = {"baseline": best_base, **paired_diff(y, S["A_linear_probe"], S[best_base], groups)}
    res["paired_A_vs_each_baseline"] = {k: paired_diff(y, S["A_linear_probe"], S[k], groups) for k in base_keys}
    if args.dataset == "pooled":
        res["by_dataset"] = {ds: {k: score(y[m], S[k][m], groups[m]) for k in ["A_linear_probe", best_base]}
                             for ds in d.dataset.unique() for m in [d.dataset.to_numpy() == ds]
                             if len(np.unique(y[m])) == 2}

    if args.target == "dep_strict":
        sp = d.spread.to_numpy(); oofB = np.zeros(len(y))
        for (tr, te), ch in zip(folds, chA):
            p = make_pipeline(StandardScaler(), PCA(min(64, len(tr) - 1), random_state=0), Ridge(alpha=100.0))
            oofB[te] = p.fit(H[tr, ch["layer"]], sp[tr]).predict(H[te, ch["layer"]])
        res["B_spread_ridge"] = dict(spearman=round(float(spearmanr(oofB, sp).statistic), 4),
                                     spearman_ci=boot_ci(sp, oofB, groups, lambda a, b: spearmanr(b, a).statistic))
        S["B_spread_ridge"] = oofB

    if args.heldout:
        L = int(pd.Series([c["layer"] for c in chA]).mode()[0])
        C = float(pd.Series([c["C"] for c in chA]).mode()[0])
        res["heldout_language"] = dict(layer=L, C=C, results=heldout_language(H, y, groups, langs, folds, L, C))

    Path("results").mkdir(exist_ok=True)
    Path(f"results/probes_{tag}.json").write_text(json.dumps(res, indent=2))
    oof = d[["prompt_id", "group", "lang", "dataset"]].copy(); oof["y"] = y
    for k, s in S.items():
        oof[k] = s
    oof.to_parquet(f"results/oof_{tag}.parquet", index=False)
    Path(f"results/layer_sweep_{tag}.json").write_text(json.dumps(layer_sweep(H, y, folds), indent=2))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in ("chosen", "results")}
                          if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
