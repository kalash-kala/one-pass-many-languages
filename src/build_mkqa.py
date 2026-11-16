"""MKQA control set: ~2000 short-answer questions x 5 languages (human translations, unchanged).

Keep examples whose English gold answer type is entity / number / number_with_unit / date
(drop unanswerable, long_answer, binary). Sample with seed 0. Prompt = the question only, in
the target language (same format as BorderLines: chat template, single user turn).
Output: data/prompts_mkqa.parquet with columns prompt_id, prompt, lang, group_id, dataset,
        gold_text, gold_aliases (per language) and gold_en.
"""
import argparse, gzip, json
from pathlib import Path

import numpy as np
import pandas as pd

LANGS = ["en", "es", "ar", "zh_cn", "ru"]
KEEP = {"entity", "number", "number_with_unit", "date"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="data/prompts_mkqa.parquet")
    args = ap.parse_args()
    ex = [json.loads(l) for l in gzip.open("data/raw/ml-mkqa/dataset/mkqa.jsonl.gz", "rt")]
    ok = []
    for e in ex:
        a = e["answers"]["en"]
        if not a or a[0]["type"] not in KEEP or not a[0].get("text"):
            continue
        if any(not e["queries"].get(l) or not e["answers"].get(l) for l in LANGS):
            continue
        ok.append(e)
    rng = np.random.default_rng(0)
    pick = [ok[i] for i in sorted(rng.choice(len(ok), size=min(args.n, len(ok)), replace=False))]
    rows = []
    for e in pick:
        g = f"mkqa_{e['example_id']}"
        en = e["answers"]["en"][0]
        for l in LANGS:
            a = e["answers"][l][0]
            rows.append(dict(prompt_id=f"{g}_{l}", prompt=e["queries"][l], lang=l, group_id=g,
                             dataset="mkqa", answer_type=en["type"], gold_en=en["text"],
                             gold_en_aliases=list(en.get("aliases", [])),
                             gold_text=a.get("text") or "", gold_aliases=list(a.get("aliases", []))))
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"eligible {len(ok)} of {len(ex)}; sampled {len(pick)} questions -> {len(df)} prompts")
    print(df.drop_duplicates("group_id").answer_type.value_counts().to_dict())
    print(df.head(5)[["lang", "prompt", "gold_text"]].to_string())


if __name__ == "__main__":
    main()
