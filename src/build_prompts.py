"""Build the open-ended BorderLines prompt table.

For every (territory, language) row in BorderLines (English control query for all 251
territories + native-language queries for claimant languages):
  1. recover the native territory name from the dataset's MC query, by matching it
     against the dataset's own per-language template (templates_q.tsv);
  2. fill our open-ended template ("Which country does XX belong to?", translated).

Output: data/prompts_borderlines.parquet  and a report of rows that could not be built.
"""
import argparse, csv, html, json, re
from pathlib import Path

import datasets
import pandas as pd

RAW = Path("data/raw/borderlines")
LETTER = re.compile(r"(?<![A-Za-z])[A-F]\)\s*")


def load_templates(paths):
    """Later files override earlier ones. Returns {code: template}."""
    out = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for r in csv.DictReader(open(p), delimiter="\t"):
            if r.get("use", "True") in ("False", "0"):
                continue
            out[r["code"]] = r["template"]
    return out


def mc_regex(tpl):
    """templates_q: 'Is XX a territory of YY or ZZ?' -> regex capturing XX and the claimant span."""
    parts = re.split(r"(XX|YY|ZZ)", tpl)
    rx = ""
    for p in parts:
        if p == "XX":
            rx += r"(?P<terr>.+?)"
        elif p == "YY":
            rx += r"(?P<claims>.+"
        elif p == "ZZ":
            rx += r".+)"  # ZZ closes the claimant span (absorbs 3+ claimants)
        else:
            lit = re.escape(p.strip()).replace(r"\ ", " ")
            rx += (r"\s*" + re.sub(r"\s+", r"\\s*", lit) + r"\s*") if lit else r"\s*"
    return re.compile(r"^\s*" + rx + r"\s*$")


def extract_territory(query, claimants, tpl):
    q = LETTER.sub("", html.unescape(query)).strip()
    claimants = [html.unescape(c) for c in claimants]
    m = mc_regex(tpl).match(q)
    if m:
        s0 = m.start("terr")
        # if the template has no words between XX and YY (e.g. Hindi), the lazy group stops
        # early; cut the territory at the first native claimant name instead.
        no_gap = re.search(r"XX\s*YY", tpl) is not None
        pos = [q.find(c, s0 + 1) for c in claimants if c and q.find(c, s0 + 1) > s0]
        if no_gap and pos and min(pos) > m.end("terr"):
            return q[s0:min(pos)].strip(" ,"), "template+claimant_cut"
        return m.group("terr").strip(" ,"), "template"
    # fallback: territory = text before the first claimant, minus the template's leading words
    first = min((q.find(c) for c in claimants if c in q), default=-1)
    return None, "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", nargs="+",
                    default=["data/templates_open.tsv", "data/templates_open_retry_final.tsv",
                             "data/templates_open_manual.tsv"])
    ap.add_argument("--drop", default="", help="comma-separated language codes to exclude")
    ap.add_argument("--out", default="data/prompts_borderlines.parquet")
    args = ap.parse_args()

    open_t = load_templates(args.templates)
    drop = set(filter(None, args.drop.split(",")))
    mc_t = {r["code"]: r["template"] for r in csv.DictReader(open(RAW / "templates_q.tsv"), delimiter="\t")}
    # BorderLines uses zhs/zht/iw in queries; map template codes if needed
    alias = {"zhs": ["zhs", "zh-CN", "zh"], "zht": ["zht", "zh-TW", "zh"], "iw": ["iw", "he"]}

    terr = datasets.load_dataset("manestay/borderlines", "territories")["train"].to_pandas()
    queries = datasets.load_dataset("manestay/borderlines", "queries")

    rows, fails = [], []
    for i, t in terr.iterrows():
        rows.append(dict(territory_idx=i, territory=t.Territory, lang="en", is_claimant_lang=False,
                         claimants=list(t.Claimants), claimants_native=list(t.Claimants),
                         territory_native=t.Territory, mc_query=t.Query,
                         prompt=open_t["en"].replace("XX", t.Territory), name_source="dataset"))
    for lang in queries:
        tpl_mc = next((mc_t[a] for a in alias.get(lang, [lang]) if a in mc_t), None)
        for r in queries[lang]:
            t = terr.iloc[r["Index_Territory"]]
            base = dict(territory_idx=r["Index_Territory"], territory=t.Territory, lang=lang,
                        is_claimant_lang=True, claimants=list(t.Claimants),
                        claimants_native=list(r["Claimants_Native"]), mc_query=r["Query_Native"])
            if lang in drop or lang not in open_t:
                fails.append({**base, "reason": "language dropped / no open template"}); continue
            if lang == "en":  # English-claimant query duplicates the control row
                fails.append({**base, "reason": "duplicate of English control row"}); continue
            if tpl_mc is None:
                fails.append({**base, "reason": "no MC template"}); continue
            name, src = extract_territory(r["Query_Native"], r["Claimants_Native"], tpl_mc)
            if name is None:
                fails.append({**base, "reason": "territory name not recovered"}); continue
            rows.append({**base, "territory_native": name, "name_source": src,
                         "prompt": open_t[lang].replace("XX", name)})

    df = pd.DataFrame(rows)
    df["prompt_id"] = df.territory.str.replace(" ", "_") + "_" + df.lang
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    pd.DataFrame(fails).to_csv(Path(args.out).with_suffix(".skipped.csv"), index=False)
    print(f"built {len(df)} prompts ({df.lang.ne('en').sum()} native), skipped {len(fails)}")
    print(pd.DataFrame(fails).groupby("reason").size() if fails else "")
    print(df.groupby("territory_idx").size().value_counts().sort_index().rename("langs/territory"))
    for lang in ["zhs", "ar", "hi", "ru", "es", "sr", "ko", "ja"]:
        s = df[df.lang == lang].head(2)
        for _, x in s.iterrows():
            print(f"[{lang}] {x.mc_query}  ->  {x.prompt}")


if __name__ == "__main__":
    main()
