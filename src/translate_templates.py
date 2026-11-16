"""Translate the open-ended question template into every BorderLines language.

Template (English): "Which country does XX belong to?"  -- XX is a placeholder that
must survive translation verbatim; it is later replaced by the dataset's own native
territory name.

Each translation is back-translated to English by the same model so a human can audit it.
Output: data/templates_open.tsv  (code, language, template, back_translation, xx_ok)
"""
import argparse, csv, json, os, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EN_TEMPLATE = "Which country does XX belong to?"

# BorderLines language codes -> language names (zhs/zht and iw are BorderLines-specific codes)
LANGS = {
    "sq": "Albanian", "ar": "Arabic", "az": "Azerbaijani", "bn": "Bengali", "bs": "Bosnian",
    "da": "Danish", "el": "Greek", "es": "Spanish", "fa": "Persian (Farsi)", "fr": "French",
    "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole", "hy": "Armenian", "id": "Indonesian",
    "is": "Icelandic", "it": "Italian", "iw": "Hebrew", "ja": "Japanese", "ka": "Georgian",
    "km": "Khmer", "ko": "Korean", "ky": "Kyrgyz", "lo": "Lao", "mg": "Malagasy",
    "mn": "Mongolian", "ms": "Malay", "my": "Burmese", "ne": "Nepali", "nl": "Dutch",
    "pt": "Portuguese", "ru": "Russian", "sl": "Slovenian", "sn": "Shona", "so": "Somali",
    "sr": "Serbian", "sw": "Swahili", "tg": "Tajik", "th": "Thai", "ti": "Tigrinya",
    "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "zhs": "Simplified Chinese", "zht": "Traditional Chinese",
}

FWD = ("Translate the following English question into {lang}. The token XX is a placeholder "
       "for a place name: keep it exactly as XX (Latin letters, unchanged) and place it where "
       "the place name belongs grammatically. Output only the translated question, nothing else.\n\n"
       "{text}")
BACK = ("Translate the following {lang} question into English. The token XX is a placeholder: "
        "keep it exactly as XX. Output only the English translation, nothing else.\n\n{text}")


def generate(model, tok, prompts, max_new_tokens=64):
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in prompts]
    enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--out", default="data/templates_open.tsv")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N languages")
    ap.add_argument("--langs", default=None, help="comma-separated subset of codes")
    ap.add_argument("--back_only", default=None,
                    help="TSV from a previous run: only back-translate its templates (no en row)")
    args = ap.parse_args()
    torch.manual_seed(0)

    codes = args.langs.split(",") if args.langs else list(LANGS)[: args.limit]
    given = None
    if args.back_only:
        rows = list(csv.DictReader(open(args.back_only), delimiter="\t"))
        given = {r["code"]: r["template"] for r in rows if r["code"] != "en"}
        codes = [c for c in codes if c in given]
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()

    if given:
        fwd = [given[c] for c in codes]
    else:
        fwd = generate(model, tok, [FWD.format(lang=LANGS[c], text=EN_TEMPLATE) for c in codes])
    back = generate(model, tok, [BACK.format(lang=LANGS[c], text=t) for c, t in zip(codes, fwd)])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["code", "language", "template", "back_translation", "xx_ok"])
        if not args.langs and not given:
            w.writerow(["en", "English", EN_TEMPLATE, EN_TEMPLATE, True])
        for c, t, b in zip(codes, fwd, back):
            w.writerow([c, LANGS[c], t.replace("\n", " "), b.replace("\n", " "),
                        t.count("XX") == 1])
    meta = {"model": args.model, "back_only_from": args.back_only, "langs": codes, "en_template": EN_TEMPLATE, "fwd_prompt": FWD,
            "back_prompt": BACK, "decoding": "greedy, max_new_tokens=64",
            "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(open(args.out).read())


if __name__ == "__main__":
    sys.exit(main())
