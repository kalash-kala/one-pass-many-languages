"""Compare regenerated results/ against a reference copy (default expected/results). Prints max |diff| per file."""
import json, sys
from pathlib import Path

ref = Path(sys.argv[1] if len(sys.argv) > 1 else "expected/results")

def walk(a, b, path=""):
    if isinstance(a, dict):
        keys = set(a) | set(b if isinstance(b, dict) else {})
        for k in keys:
            if not isinstance(b, dict) or k not in a or k not in b:
                yield path + "/" + str(k), float("inf")
            else:
                yield from walk(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            yield path, float("inf"); return
        for i, (x, y) in enumerate(zip(a, b)):
            yield from walk(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        yield path, abs(a - b)
    else:
        yield path, 0.0 if a == b else float("inf")

bad = 0
for f in sorted(ref.glob("*.json")):
    g = Path("results") / f.name
    if not g.exists():
        print(f"MISSING  {f.name}"); bad += 1; continue
    diffs = list(walk(json.loads(f.read_text()), json.loads(g.read_text())))
    worst = max(diffs, key=lambda x: x[1]) if diffs else ("", 0.0)
    flag = "OK " if worst[1] == 0 else "DIFF"
    bad += worst[1] != 0
    print(f"{flag} {f.name}: max|diff|={worst[1]:.3g} {worst[0] if worst[1] else ''}")
for name in ["main_table.md"]:
    g = Path("results") / name
    if not g.exists():
        print(f"MISSING  {name}"); bad += 1; continue
    same = (ref / name).read_text() == g.read_text()
    print(("OK " if same else "DIFF"), name); bad += not same
print("ALL IDENTICAL" if bad == 0 else f"{bad} file(s) differ")
