"""Paired bootstrap for macro-F1 deltas, rebuilt from per-item run records.

Mirrors src/aggregate.py: the C3 score is the present count, the label is
searchsorted(cuts, score, side="right"), and records without parsed items are dropped.
Refuses to write output unless it reproduces numbers.json's macro_f1, qwk and
sev_missed for every cell, and the paper's [removed]-exclusion figures (53 posts,
kappa_w C3 fitted - C2 = -0.051 on both DepSign models).
"""
import csv
import glob
import json
import os
import sys
import numpy as np

RUNS = "../runs" if os.path.isdir("../runs") else "../runs_scrubbed"
N = json.load(open("numbers.json"))
LABELS = {"depseverity": ["minimum", "mild", "moderate", "severe"],
          "depsign": ["not depression", "moderate", "severe"]}
TAG = {"Qwen3.5-9B": "ollama-qwen3.5-9b", "DeepSeek-V4.1-Flash": "deepseek-deepseek-flash",
       "Claude-Sonnet-5": "anthropic-claude-sonnet-5"}
B, SEED = 4000, 0
COMPARISONS = (("C2", "C1"), ("C3 fitted", "C2"), ("C3 a priori", "C2"))


def load(corpus, cond, model):
    pre = "" if corpus == "depseverity" else "depsign_"
    path = f"{RUNS}/{pre}{cond}_{TAG[model]}_test_seed0.jsonl"
    return {r["id"]: r for r in map(json.loads, open(path))}


def macro_f1(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    denom = cm.sum(-1) + cm.sum(-2)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean(-1)


def qwk(cm):
    k = cm.shape[-1]
    i, j = np.indices((k, k))
    w = (i - j) ** 2 / (k - 1) ** 2
    n = cm.sum((-2, -1), keepdims=True)
    exp = cm.sum(-1)[..., :, None] * cm.sum(-2)[..., None, :] / n
    return 1 - (w * cm).sum((-2, -1)) / (w * exp).sum((-2, -1))


def confusion(g, p, k, idx=None):
    if idx is None:
        return np.bincount(k * g + p, minlength=k * k).reshape(k, k)
    flat = k * g[idx] + p[idx] + np.arange(len(idx))[:, None] * k * k
    return np.bincount(flat.ravel(), minlength=len(idx) * k * k).reshape(len(idx), k, k)


def predictions(corpus, model):
    lab = {s: i for i, s in enumerate(LABELS[corpus])}
    recs = {c: load(corpus, c, model) for c in ("C1", "C2", "C3")}
    gold = {i: lab[r["gold"]] for i, r in recs["C1"].items()}
    preds = {c: {i: lab[r["pred"]] for i, r in recs[c].items()} for c in ("C1", "C2")}
    score = {i: sum(v["status"] == "present" for v in r["items"].values())
             for i, r in recs["C3"].items() if r.get("items")}
    for reg in ("fitted", "a priori"):
        cuts = np.asarray(N[corpus]["cells"][f"{model}|C3 {reg}"]["cutoffs"])
        preds[f"C3 {reg}"] = {i: int(np.searchsorted(cuts, s, side="right"))
                              for i, s in score.items()}
    return gold, preds


def paired(gold, preds, a, b, k, exclude=frozenset()):
    ids = sorted((set(preds[a]) & set(preds[b])) - exclude)
    g = np.array([gold[i] for i in ids])
    pa, pb = (np.array([preds[x][i] for i in ids]) for x in (a, b))
    idx = np.random.default_rng(SEED).integers(0, len(ids), (B, len(ids)))
    ca, cb = confusion(g, pa, k, idx), confusion(g, pb, k, idx)
    d = float(macro_f1(confusion(g, pa, k)) - macro_f1(confusion(g, pb, k)))
    dk = float(qwk(confusion(g, pa, k)) - qwk(confusion(g, pb, k)))
    lo, hi = np.percentile(macro_f1(ca) - macro_f1(cb), [2.5, 97.5])
    klo, khi = np.percentile(qwk(ca) - qwk(cb), [2.5, 97.5])
    return {"d": d, "ci": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0),
            "n": len(ids), "kw_d": dk, "kw_ci": [float(klo), float(khi)]}


def removed_test_ids():
    csv.field_size_limit(sys.maxsize)
    paths = glob.glob("../data/**/depsign_split.csv", recursive=True)
    if not paths:
        return None
    [path] = paths
    with open(path, newline="") as f:
        return frozenset(int(r["id"]) for r in csv.DictReader(f)
                         if r["split"] == "test"
                         and ("[removed]" in r["text"] or "[deleted]" in r["text"]))


out, maxdiff = {}, 0.0
for corpus in ("depseverity", "depsign"):
    k = len(LABELS[corpus])
    for model in sorted({c.split("|")[0] for c in N[corpus]["cells"]}):
        gold, preds = predictions(corpus, model)
        for name, pd in preds.items():            # validate each condition on its own ids
            ids = sorted(pd)
            cm = confusion(np.array([gold[i] for i in ids]), np.array([pd[i] for i in ids]), k)
            ref = N[corpus]["cells"][f"{model}|{name}"]
            maxdiff = max(maxdiff, abs(float(macro_f1(cm)) - ref["macro_f1"]),
                          abs(float(qwk(cm)) - ref["qwk"]))
            assert int(cm[-1].sum() - cm[-1, -1]) == ref["sev_missed"], (corpus, model, name)
        for a, b in COMPARISONS:
            key = f"{model}|{a} - {b}"
            res = paired(gold, preds, a, b, k)
            out.setdefault(corpus, {})[key] = res
            ref = N[corpus]["deltas"][key]
            print(f"{corpus[:6]} {key:38} n={res['n']} F1 {res['d']:+.3f} "
                  f"[{res['ci'][0]:+.3f},{res['ci'][1]:+.3f}] {'SIG' if res['sig'] else '   '}"
                  f" | kw CI [{res['kw_ci'][0]:+.3f},{res['kw_ci'][1]:+.3f}]"
                  f" published [{ref['ci'][0]:+.3f},{ref['ci'][1]:+.3f}]")
print(f"\nvalidation: max |recomputed - numbers.json| = {maxdiff:.2e}")
assert maxdiff < 1e-9, "reconstruction does not match numbers.json; not writing output"

# Sensitivity: DepSign without [removed]/[deleted] placeholder posts (paper Sec. V-D).
excl = removed_test_ids()
if excl is None:
    sys.exit("\ncorpus not found: skipping the [removed] sensitivity; macro_f1_deltas.json not "
             "rewritten (the shipped file includes it)")
print(f"\n[removed]/[deleted] DepSign test posts: {len(excl)} (paper: 53)")
assert len(excl) == 53, "placeholder set differs from the paper's; not writing output"
k = len(LABELS["depsign"])
for model in sorted({c.split("|")[0] for c in N["depsign"]["cells"]}):
    gold, preds = predictions("depsign", model)
    for a, b in COMPARISONS[1:]:
        key = f"{model}|{a} - {b}"
        res = paired(gold, preds, a, b, k, exclude=excl)
        out.setdefault("depsign_excluding_removed", {})[key] = res
        print(f"excl   {key:38} n={res['n']} F1 {res['d']:+.3f} "
              f"[{res['ci'][0]:+.3f},{res['ci'][1]:+.3f}] {'SIG' if res['sig'] else '   '}"
              f" | kw {res['kw_d']:+.3f}")
for model in ("DeepSeek-V4.1-Flash", "Claude-Sonnet-5"):
    got = out["depsign_excluding_removed"][f"{model}|C3 fitted - C2"]["kw_d"]
    assert abs(got - (-0.051)) < 0.0005, f"{model}: kw after exclusion {got:+.4f} != paper -0.051"
json.dump(out, open("macro_f1_deltas.json", "w"), indent=1)
print("wrote macro_f1_deltas.json")
