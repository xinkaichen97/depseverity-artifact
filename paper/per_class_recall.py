"""Per-class recall for every test cell (extended-version appendix).

Needs only the per-post run records (../runs, or ../runs_scrubbed in the artifact) and
numbers.json, whose frozen thresholds it applies. Mirrors src/aggregate.py like
macro_f1_boot.py does: the C3/C4 score is the present count, the label is
searchsorted(cuts, score, side="right"), and records whose statuses cannot be recovered are
dropped. Refuses to write output unless every cell reproduces numbers.json's kappa_w and
severe-missed count. Run from the paper directory.
"""
import json
import os
import re

import numpy as np

RUNS = "../runs" if os.path.isdir("../runs") else "../runs_scrubbed"
N = json.load(open("numbers.json"))
LABELS = {"depseverity": ["minimum", "mild", "moderate", "severe"],
          "depsign": ["not depression", "moderate", "severe"]}
TAG = {"Qwen3.5-9B": "ollama-qwen3.5-9b", "DeepSeek-V4.1-Flash": "deepseek-deepseek-flash",
       "Claude-Sonnet-5": "anthropic-claude-sonnet-5"}
N_ITEMS = {"C3": 9, "C4": 21, "C2S counted": 9}


def load(corpus, cond, model):
    pre = "" if corpus == "depseverity" else "depsign_"
    path = f"{RUNS}/{pre}{cond}_{TAG[model]}_test_seed0.jsonl"
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else None


def recover(raw, n):
    """Mirrors aggregate.recover_items: statuses only, and only if every item is found."""
    out = {}
    for m in re.finditer(r'"(\w+)"\s*:\s*\{', raw):
        st = re.search(r'"status"\s*:\s*"(present|absent|unclear)"', raw[m.end():])
        if st:
            out[m.group(1)] = {"status": st.group(1)}
    return out if len(out) == n else None


def fill_omitted(r, n):
    """Mirrors aggregate.fill_omitted: a valid response missing exactly one item scores it unclear."""
    err = r.get("parse_error") or ""
    if not err.startswith("missing_items: ") or "," in err:
        return None
    m = re.search(r"\{.*\}", r.get("raw") or "", re.S)
    try:
        obj = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    out = {k: {"status": v} if isinstance(v, str) else v for k, v in obj.items()}
    if len(out) != n - 1 or any(v.get("status") not in ("present", "absent", "unclear")
                                for v in out.values()):
        return None
    out[err.split(": ", 1)[1]] = {"status": "unclear"}
    return out


def qwk(g, p, k):
    cm = np.bincount(k * g + p, minlength=k * k).reshape(k, k).astype(float)
    i, j = np.indices((k, k))
    w = (i - j) ** 2 / (k - 1) ** 2
    exp = cm.sum(1)[:, None] * cm.sum(0)[None, :] / cm.sum()
    return 1 - (w * cm).sum() / (w * exp).sum()


out = {}
for corpus, labels in LABELS.items():
    lab = {s: i for i, s in enumerate(labels)}
    k = len(labels)
    out[corpus] = {"labels": labels, "cells": {}}
    for model in TAG:
        for cond in ("C1", "C2", "C3", "C4", "C2S", "C2S counted"):
            # "C2S counted" re-reads the C2S records and counts their checklist lines.
            recs = load(corpus, cond.split()[0], model)
            if recs is None:
                continue
            for reg in (("fitted", "a priori") if cond in N_ITEMS else (None,)):
                key = f"{model}|{cond}" + (f" {reg}" if reg else "")
                if reg is None:
                    pairs = [(lab[r["gold"]], lab[r["pred"]]) for r in recs if r.get("pred")]
                else:
                    cuts = np.asarray(N[corpus]["cells"][key]["cutoffs"])
                    pairs = []
                    for r in recs:
                        items = (r.get("items") or recover(r.get("raw") or "", N_ITEMS[cond])
                                 or fill_omitted(r, N_ITEMS[cond]))
                        if items is None:
                            continue
                        s = sum(v["status"] == "present" for v in items.values())
                        pairs.append((lab[r["gold"]], int(np.searchsorted(cuts, s, side="right"))))
                g, p = (np.array(x) for x in zip(*pairs))
                ref = N[corpus]["cells"][key]
                sev_missed = int(((g == k - 1) & (p < k - 1)).sum())
                assert abs(qwk(g, p, k) - ref["qwk"]) < 1e-9, (corpus, key)
                assert sev_missed == ref["sev_missed"], (corpus, key, sev_missed)
                out[corpus]["cells"][key] = {
                    "recall": [float((p[g == c] == c).mean()) for c in range(k)],
                    "n": [int((g == c).sum()) for c in range(k)],
                    "pred_share": [float((p == c).mean()) for c in range(k)]}

json.dump(out, open("per_class_recall.json", "w"), indent=1)
for corpus in out:
    print(corpus, out[corpus]["labels"])
    for key, v in out[corpus]["cells"].items():
        print(f"  {key:34} " + "  ".join(f"{r:.2f}" for r in v["recall"]))
