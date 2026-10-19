#!/usr/bin/env python3
"""Check a decision-model JSONL file against the Unsloth / Laya dataset rules before training.
Rules (Unsloth docs, "Dataset Guide"): every row has state, questions, gold; every question has a gold
answer; choice gold is a key of criteria; noul gold is true/false; score gold is a 0-based level index.
Also prints per-label counts so you can see thin labels before you spend GPU time."""
import json, sys
from collections import Counter

def check(path):
    errors, counts = [], Counter()
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        q = row.get("questions"); g = row.get("gold")
        q = json.loads(q) if isinstance(q, str) else q
        g = json.loads(g) if isinstance(g, str) else g
        if row.get("state") in (None, "") or not isinstance(q, dict) or not isinstance(g, dict):
            errors.append(f"{path}:{n}: needs state, questions and gold"); continue
        for name, spec in q.items():
            t, crit = spec.get("type"), spec.get("criteria")
            if name not in g:
                errors.append(f"{path}:{n}: '{name}' has no gold"); continue
            ans = g[name]["label"] if isinstance(g[name], dict) else g[name]
            if t == "choice" and (not isinstance(crit, dict) or str(ans) not in map(str, crit)):
                errors.append(f"{path}:{n}: '{name}' gold {ans!r} is not a criteria key")
            elif t == "noul" and not (isinstance(ans, bool) or str(ans).lower() in ("true", "false")):
                errors.append(f"{path}:{n}: '{name}' noul gold must be true/false")
            elif t == "score" and not (isinstance(crit, list) and str(ans).isdigit() and int(ans) < len(crit)):
                errors.append(f"{path}:{n}: '{name}' score gold must be a level index < {len(crit or [])}")
            elif t not in ("choice", "noul", "score"):
                errors.append(f"{path}:{n}: '{name}' unknown type {t!r}")
            counts[(name, str(ans).lower())] += 1
    return errors, counts

if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        errs, counts = check(p)
        print(f"== {p}: {sum(counts.values())} decisions, {len(errs)} errors")
        for e in errs[:50]:
            print("  ", e)
        for (q, label), c in sorted(counts.items()):
            print(f"   {q:>12} = {label:<12} {c}")
        bad += len(errs)
    sys.exit(1 if bad else 0)
