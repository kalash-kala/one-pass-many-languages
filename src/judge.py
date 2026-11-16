"""Judge (Llama-3.3-70B-Instruct): map each free-text BorderLines answer to a canonical label.

Labels:  one of the territory's claimants (English canonical name)
         "DISPUTED"  - answer says it is disputed / names several claimants without choosing
         "REFUSAL"   - declines to answer
         "OTHER:<country>" - names a single country that is not a listed claimant
         "UNCLEAR"   - anything else
Reads cache/<model>/borderlines/chunk_*.jsonl + the prompt table; writes
cache/<model>/borderlines/judged.parquet (resumable per model).
"""
import argparse, glob, json, re, sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_ID = "meta-llama/Llama-3.3-70B-Instruct"

PROMPT = """You are labelling answers from a language model. The model was asked, in {lang_name}, which country the territory "{territory}" belongs to. The claimant countries are: {claimants}.

Model answer (may be in any language):
<<<
{answer}
>>>

Classify the answer's position on which country the territory belongs to. Choose exactly one label:
- one of the claimant names exactly as written above, if the answer assigns the territory to that country (even if it mentions the dispute);
- "DISPUTED" if the answer says it is disputed or lists several claimants without assigning it to one;
- "REFUSAL" if the answer declines to answer;
- "OTHER:<country name in English>" if it assigns the territory to a single country that is not in the claimant list;
- "UNCLEAR" if none of the above applies.

Respond with JSON only: {{"label": "...", "english_gloss": "<one-sentence English summary of the answer>"}}"""


def parse(text, claimants):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, None, "no_json"
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None, "bad_json"
    lab = str(d.get("label", "")).strip()
    canon = {c.lower(): c for c in claimants}
    if lab.lower() in canon:
        return canon[lab.lower()], d.get("english_gloss"), "ok"
    if lab in ("DISPUTED", "REFUSAL", "UNCLEAR") or lab.startswith("OTHER:"):
        return lab, d.get("english_gloss"), "ok"
    return None, d.get("english_gloss"), f"invalid_label:{lab}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama", "qwen", "gemma"])
    ap.add_argument("--prompts", default="data/prompts_borderlines.parquet")
    ap.add_argument("--root", default="cache")
    ap.add_argument("--subdir", default="borderlines")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    langnames = pd.read_csv("data/templates_open.tsv", sep="\t").set_index("code").language.to_dict()
    pr = pd.read_parquet(args.prompts).set_index("prompt_id")
    tok = AutoTokenizer.from_pretrained(JUDGE_ID); tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(JUDGE_ID, dtype=torch.bfloat16, device_map="auto").eval()

    for m in args.models:
        d = Path(args.root) / m / args.subdir
        outp = d / "judged.parquet"
        files = sorted(glob.glob(str(d / "chunk_*.jsonl")))
        if outp.exists() or not files:
            print(f"[{m}] skip ({'done' if outp.exists() else 'no answers'})"); continue
        ans = pd.concat([pd.read_json(f, lines=True) for f in files]).set_index("prompt_id")
        ans = ans.join(pr[["territory", "claimants", "lang"]])
        prompts = [PROMPT.format(lang_name=langnames.get(r.lang, r.lang), territory=r.territory,
                                 claimants=", ".join(r.claimants), answer=r.answer[:1500])
                   for r in ans.itertuples()]
        raw = []
        for b in range(0, len(prompts), args.batch_size):
            chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True) for p in prompts[b:b + args.batch_size]]
            enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=96, do_sample=False, temperature=None,
                                   top_p=None, pad_token_id=tok.pad_token_id)
            raw += [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True) for x in g]
            print(f"[{m}] judged {len(raw)}/{len(prompts)}", flush=True)
        parsed = [parse(t, c) for t, c in zip(raw, ans.claimants)]
        ans["judge_raw"] = raw
        ans["label"] = [p[0] for p in parsed]
        ans["english_gloss"] = [p[1] for p in parsed]
        ans["judge_status"] = [p[2] for p in parsed]
        ans.reset_index().drop(columns=["token_logprobs"]).to_parquet(outp, index=False)
        print(f"[{m}] status:", ans.judge_status.value_counts().to_dict())
        print(f"[{m}] labels:", ans.label.map(lambda x: str(x).split(":")[0]).value_counts().to_dict())


if __name__ == "__main__":
    sys.exit(main())
