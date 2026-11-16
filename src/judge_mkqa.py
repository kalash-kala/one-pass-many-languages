"""MKQA judge (Llama-3.3-70B-Instruct, bf16): per question, all 5 language answers at once.

Output per question: correctness per language (correct / incorrect / refusal) and a grouping
of languages whose answers refer to the same thing. Also an alias string-match correctness
(gold text + aliases, per-language and English) as a judge-independent check.
Writes cache/<model>/mkqa/judged.parquet (one row per prompt) for each model.
"""
import argparse, glob, json, re, sys, unicodedata
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_ID = "meta-llama/Llama-3.3-70B-Instruct"
LANGS = ["en", "es", "ar", "zh_cn", "ru"]

PROMPT = """A language model was asked the same question in five languages. Judge its answers.

Question (English): {q}
Gold answer: {gold} (aliases: {aliases})

Answers:
{answers}

Tasks:
1. For each language code, mark the answer "correct" (matches the gold answer or an alias, in any language/script), "incorrect", or "refusal" (declines or says it does not know).
2. Group the language codes whose answers refer to the same entity/value (regardless of language or script, and regardless of correctness). Every code must appear in exactly one group; refusals go in their own group(s).

Respond with JSON only:
{{"correct": {{"en": "...", "es": "...", "ar": "...", "zh_cn": "...", "ru": "..."}}, "groups": [["en", "..."], ["..."]]}}"""


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"[\s\.\,\-\'\"·]+", " ", s).strip()


def alias_match(answer, golds):
    a = norm(answer)
    return any(g and norm(g) in a for g in golds)


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, "no_json"
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "bad_json"
    c, g = d.get("correct", {}), d.get("groups", [])
    flat = [x for grp in g for x in grp]
    if set(c) != set(LANGS) or sorted(flat) != sorted(LANGS):
        return d, "invalid_structure"
    return d, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama", "qwen", "gemma"])
    ap.add_argument("--batch_size", type=int, default=12)
    args = ap.parse_args()
    pr = pd.read_parquet("data/prompts_mkqa.parquet")
    tok = AutoTokenizer.from_pretrained(JUDGE_ID); tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(JUDGE_ID, dtype=torch.bfloat16, device_map="auto").eval()

    for m in args.models:
        d = Path(f"cache/{m}/mkqa"); outp = d / "judged.parquet"
        if outp.exists() or not (d / "DONE").exists():
            print(f"[{m}] skip"); continue
        ans = pd.concat([pd.read_json(f, lines=True) for f in sorted(glob.glob(str(d / "chunk_*.jsonl")))])
        df = pr.merge(ans[["prompt_id", "answer", "n_tokens", "truncated", "first_token_entropy"]], on="prompt_id")
        groups = list(df.groupby("group_id", sort=False))
        prompts = []
        for gid, g in groups:
            g = g.set_index("lang")
            prompts.append(PROMPT.format(q=g.loc["en", "prompt"], gold=g.loc["en", "gold_en"],
                                         aliases="; ".join(g.loc["en", "gold_en_aliases"]) or "none",
                                         answers="\n".join(f"[{l}] {g.loc[l, 'answer'][:400]}" for l in LANGS)))
        raw = []
        for b in range(0, len(prompts), args.batch_size):
            chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True) for p in prompts[b:b + args.batch_size]]
            enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=160, do_sample=False, temperature=None,
                                     top_p=None, pad_token_id=tok.pad_token_id)
            raw += [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True) for x in out]
            if len(raw) % (args.batch_size * 20) == 0:
                print(f"[{m}] judged {len(raw)}/{len(prompts)}", flush=True)
        recs = []
        for (gid, g), r in zip(groups, raw):
            dct, status = parse(r)
            gmap = {}
            if status == "ok":
                for k, grp in enumerate(dct["groups"]):
                    for l in grp:
                        gmap[l] = k
            for row in g.itertuples():
                golds = [row.gold_text, *row.gold_aliases, row.gold_en, *row.gold_en_aliases]
                recs.append(dict(prompt_id=row.prompt_id, judge_status=status, judge_raw=r,
                                 judge_correct=(dct or {}).get("correct", {}).get(row.lang) if status == "ok" else None,
                                 answer_group=gmap.get(row.lang),
                                 alias_correct=alias_match(row.answer, golds)))
        J = df.merge(pd.DataFrame(recs), on="prompt_id")
        J.to_parquet(outp, index=False)
        print(f"[{m}] status:", J.drop_duplicates("group_id").judge_status.value_counts().to_dict())
        ok = J[J.judge_status == "ok"]
        agree = ((ok.judge_correct == "correct") == ok.alias_correct).mean()
        print(f"[{m}] judge-vs-alias correctness agreement: {agree:.3f}; judge correct rate "
              f"{(ok.judge_correct == 'correct').mean():.3f}; alias correct rate {ok.alias_correct.mean():.3f}")


if __name__ == "__main__":
    sys.exit(main())
