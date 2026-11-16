"""Per-model labels from judged answers.

Per territory (group), over all languages it was asked in:
  * dep_strict (PRIMARY): 1 if the canonical labels are not all identical. A switch to
    DISPUTED / REFUSAL / OTHER / UNCLEAR counts as a change.
  * dep_claimant: 1 if the answers name more than one distinct claimant (only answers that
    pick a claimant are considered; NaN if fewer than 2 such answers).
  * spread: 1 - (share of languages giving the majority label).
Row-level: row_diff = this language's label differs from the territory's majority label.
Territories asked in only one language are dropped (nothing to compare).
"""
import argparse, json
from pathlib import Path

import pandas as pd

SPECIAL = ("DISPUTED", "REFUSAL", "UNCLEAR")


ALIAS = {"china": "People's Republic of China", "prc": "People's Republic of China",
         "taiwan": "Republic of China", "roc": "Republic of China"}


def normalize(label, claimants, gloss):
    """Map judge output to a claimant / DISPUTED / REFUSAL / OTHER:<x> / UNCLEAR.
    Fixes (counted in label_fix): OTHER:<claimant variant>, and invalid labels the judge wrote
    in the answer's script -> recovered from the English gloss iff exactly one claimant is named."""
    cl = list(claimants)
    if isinstance(label, str) and label.startswith("OTHER:"):
        x = label[6:].strip()
        if ALIAS.get(x.lower()) in cl:
            return ALIAS[x.lower()], "other_alias"
        hits = [c for c in cl if x.lower() == c.lower() or x.lower() in c.lower() or c.lower() in x.lower()]
        if len(hits) == 1:
            return hits[0], "other_substring"
        return "OTHER:" + x, "none"
    if label is None or (isinstance(label, float) and pd.isna(label)) or label == "None":
        g = (gloss or "").lower()
        hits = [c for c in cl if c.lower() in g]
        # drop shorter names contained in a longer hit (e.g. Sudan inside South Sudan)
        hits = [c for c in hits if not any(c != h and c.lower() in h.lower() for h in hits)]
        if len(hits) == 1 and "disputed" not in g:
            return hits[0], "gloss_recovered"
        return "UNCLEAR", "invalid_to_unclear"
    return label, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama", "qwen", "gemma"])
    ap.add_argument("--prompts", default="data/prompts_borderlines.parquet")
    args = ap.parse_args()
    pr = pd.read_parquet(args.prompts)
    summary = {}
    for m in args.models:
        jp = Path(f"cache/{m}/borderlines/judged.parquet")
        if not jp.exists():
            print(f"[{m}] no judged.parquet"); continue
        j = pd.read_parquet(jp)[["prompt_id", "answer", "label", "english_gloss", "judge_status",
                                 "first_token_entropy", "n_tokens", "truncated"]]
        d = pr.merge(j, on="prompt_id")
        fixed = [normalize(l, c, g) for l, c, g in zip(d.label, d.claimants, d.english_gloss)]
        d["label_raw"] = d.label
        d["label"] = [f[0] for f in fixed]
        d["label_fix"] = [f[1] for f in fixed]
        d["is_claimant_answer"] = ~d.label.isin(SPECIAL) & ~d.label.str.startswith("OTHER:")
        g = d.groupby("territory_idx")
        nl = g.lang.transform("size")
        d = d[nl >= 2].copy()
        g = d.groupby("territory_idx")
        maj = g.label.agg(lambda s: s.value_counts().index[0])
        per = pd.DataFrame({
            "n_langs": g.size(),
            "dep_strict": g.label.agg(lambda s: int(s.nunique() > 1)),
            "dep_claimant": g.apply(lambda x: (int(x.loc[x.is_claimant_answer, "label"].nunique() > 1)
                                               if x.is_claimant_answer.sum() >= 2 else float("nan"))),
            "spread": g.label.agg(lambda s: 1 - s.value_counts().iloc[0] / len(s)),
            "majority": maj,
        })
        d = d.join(per, on="territory_idx")
        d["row_diff"] = (d.label != d.majority).astype(int)
        out = Path(f"cache/{m}/borderlines/labels.parquet")
        d.to_parquet(out, index=False)
        s = dict(n_rows=len(d), n_territories=len(per),
                 dep_strict_rate=round(per.dep_strict.mean(), 3),
                 dep_claimant_rate=round(per.dep_claimant.mean(), 3),
                 dep_claimant_n=int(per.dep_claimant.notna().sum()),
                 row_diff_rate=round(d.row_diff.mean(), 3),
                 label_dist=d.label.map(lambda x: x.split(":")[0] if x.startswith("OTHER") or x in SPECIAL
                                        else "CLAIMANT").value_counts().to_dict(),
                 judge_status=d.judge_status.value_counts().to_dict(),
                 label_fix=d.label_fix.value_counts().to_dict())
        summary[m] = s
        print(m, json.dumps(s, indent=1))
    Path("results").mkdir(exist_ok=True)
    if summary:
        Path("results/class_balance.json").write_text(json.dumps(summary, indent=2))
    mkqa_labels(args.models)


def mkqa_labels(models):
    """MKQA: label = judge's answer group id per language (refusals are their own groups).
    dep_strict = more than one group across the 5 languages; spread = 1 - largest group share.
    Questions whose judge output failed validation are dropped (count reported)."""
    summary = {}
    for m in models:
        jp = Path(f"cache/{m}/mkqa/judged.parquet")
        if not jp.exists():
            print(f"[{m}] no mkqa judged.parquet"); continue
        d = pd.read_parquet(jp)
        n_q = d.group_id.nunique()
        d = d[d.judge_status == "ok"].copy()
        d["label"] = d.answer_group.astype(int).astype(str)
        g = d.groupby("group_id")
        per = pd.DataFrame({"n_langs": g.size(),
                            "dep_strict": g.label.agg(lambda s: int(s.nunique() > 1)),
                            "spread": g.label.agg(lambda s: 1 - s.value_counts().iloc[0] / len(s)),
                            "majority": g.label.agg(lambda s: s.value_counts().index[0])})
        d = d.join(per, on="group_id")
        d["row_diff"] = (d.label != d.majority).astype(int)
        d.to_parquet(f"cache/{m}/mkqa/labels.parquet", index=False)
        summary[m] = dict(n_questions=int(n_q), n_valid=int(len(per)), n_rows=len(d),
                          dep_strict_rate=round(per.dep_strict.mean(), 3),
                          row_diff_rate=round(d.row_diff.mean(), 3),
                          judge_correct_rate=round((d.judge_correct == "correct").mean(), 3),
                          alias_correct_rate=round(d.alias_correct.mean(), 3),
                          judge_alias_agreement=round(((d.judge_correct == "correct") == d.alias_correct).mean(), 3),
                          refusal_rate=round((d.judge_correct == "refusal").mean(), 3))
        print("mkqa", m, json.dumps(summary[m], indent=1))
    if summary:
        Path("results/class_balance_mkqa.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
