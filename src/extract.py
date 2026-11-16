"""Extraction (the only subject-model GPU step).

For every prompt in a prompt table (parquet with columns prompt_id, prompt, ...):
  * chat-templated user turn containing ONLY the question (no system prompt added by us;
    the model's own template defaults are kept and recorded);
  * hidden state at the LAST PROMPT TOKEN for every layer (embeddings + each block), float16;
  * greedy answer (max_new_tokens) + per-token log-probs and first-token entropy
    (inputs for the single-pass confidence baseline).

Resumable: writes chunk_XXXXX.npz + chunk_XXXXX.jsonl and skips chunks already on disk.
Usage: python src/extract.py --model llama --prompts data/prompts_borderlines.parquet --out cache/llama/borderlines
"""
import argparse, json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-3-12b-it",
}


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N prompts")
    ap.add_argument("--save_dtype", default="float16", choices=["float16", "float32"],
                    help="float32 for Gemma: one channel exceeds the float16 range in later layers")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.prompts)
    if args.limit:
        df = df.groupby("lang", sort=False).head(1).head(args.limit) if args.limit < len(df) else df
    df = df.reset_index(drop=True)

    hf_id = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    meta = dict(model=args.model, hf_id=hf_id, class_=type(model).__name__,
                prompt_format="chat template, single user turn = question only, add_generation_prompt=True",
                example_chat=chat(tok, df.prompt.iloc[0]), decoding="greedy",
                max_new_tokens=args.max_new_tokens, seed=0, save_dtype=args.save_dtype, n_prompts=len(df),
                torch=torch.__version__, transformers=transformers.__version__,
                python=platform.python_version(), host=platform.node(),
                git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                          text=True).stdout.strip() or None)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    eos = list({tok.eos_token_id, *(model.generation_config.eos_token_id
                                    if isinstance(model.generation_config.eos_token_id, list)
                                    else [model.generation_config.eos_token_id])} - {None})
    t0 = time.time()
    for c0 in range(0, len(df), args.chunk):
        stem = out / f"chunk_{c0 // args.chunk:05d}"
        if stem.with_suffix(".npz").exists() and stem.with_suffix(".jsonl").exists():
            continue
        sub = df.iloc[c0:c0 + args.chunk]
        hs_all, recs = [], []
        for b0 in range(0, len(sub), args.batch_size):
            bt = sub.iloc[b0:b0 + args.batch_size]
            enc = tok([chat(tok, p) for p in bt.prompt], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                   temperature=None, top_p=None, top_k=None,
                                   output_hidden_states=True, output_scores=True,
                                   return_dict_in_generate=True, pad_token_id=tok.pad_token_id,
                                   eos_token_id=eos)
            # prefill step: tuple over layers of (B, T, D); last position = last prompt token
            hs = torch.stack([h[:, -1, :] for h in g.hidden_states[0]], dim=1)  # (B, L+1, D)
            hs_all.append(hs.float().cpu().numpy().astype(args.save_dtype))
            gen = g.sequences[:, enc.input_ids.shape[1]:]
            logp = torch.stack([torch.log_softmax(s.float(), -1) for s in g.scores], 1)  # (B,N,V)
            first = logp[:, 0]
            ent0 = -(first.exp() * first).sum(-1)
            tok_lp = logp.gather(-1, gen.unsqueeze(-1)).squeeze(-1)
            for i, (_, r) in enumerate(bt.iterrows()):
                ids = gen[i].tolist()
                n = next((k for k, t in enumerate(ids) if t in eos or t == tok.pad_token_id), len(ids))
                recs.append(dict(prompt_id=r.prompt_id, answer=tok.decode(ids[:n], skip_special_tokens=True).strip(),
                                 n_tokens=n, truncated=n == len(ids),
                                 token_logprobs=[round(x, 5) for x in tok_lp[i, :n].tolist()],
                                 first_token_entropy=float(ent0[i])))
        np.savez(stem.with_suffix(".npz"), hidden=np.concatenate(hs_all),
                 prompt_id=np.array(sub.prompt_id.tolist()))
        with open(stem.with_suffix(".jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{args.model}] chunk {c0 // args.chunk} done ({c0 + len(sub)}/{len(df)}) "
              f"{time.time() - t0:.0f}s", flush=True)
    H = np.concatenate([np.load(f)["hidden"] for f in sorted(out.glob("chunk_*.npz"))])
    assert np.isfinite(H).all(), f"non-finite hidden states: {(~np.isfinite(H)).sum()}"
    (out / "DONE").write_text(str(time.time() - t0))


if __name__ == "__main__":
    sys.exit(main())
