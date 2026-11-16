"""Main-text figure: layer sweep, primary label, BorderLines (solid) + MKQA (dashed). Reads results/ only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"llama": "#1f77b4", "qwen": "#ff7f0e", "gemma": "#2ca02c"}
N = {"llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B", "gemma": "Gemma-3-12B"}
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for m in C:
    for ds, ls in [("borderlines", "-"), ("mkqa", "--")]:
        s = json.loads(Path(f"results/layer_sweep_{m}_{ds}_dep_strict.json").read_text())
        n = len(s) - 1
        ax.plot([x["layer"] / n for x in s], [x["auroc"] for x in s], ls=ls, color=C[m], lw=1.4,
                label=f"{N[m]}" if ds == "borderlines" else None)
ax.axhline(0.5, color="grey", lw=0.8, ls=":")
ax.plot([], [], "k-", label="BorderLines"); ax.plot([], [], "k--", label="MKQA")
ax.set_xlabel("relative depth (layer / #layers)"); ax.set_ylabel("AUROC")
ax.legend(fontsize=7, ncol=2, loc="lower right"); fig.tight_layout()
for e in ["pdf", "png"]:
    fig.savefig(f"figures/layer_sweep_main.{e}", dpi=200)
