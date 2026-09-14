"""Dump every number the paper cites, so nothing is transcribed by hand."""
import json
import numpy as np

import aggregate as A
import conditions as C
import data as D
import data_depsign as DS
import evaluate as E

MODELS = ["ollama:qwen3.5:9b", "deepseek:deepseek-flash", "anthropic:claude-sonnet-5"]
SHORT = {"ollama:qwen3.5:9b": "Qwen3.5-9B", "deepseek:deepseek-flash": "DeepSeek-V4.1-Flash",
         "anthropic:claude-sonnet-5": "Claude-Sonnet-5"}
CORPORA = {"depseverity": (D, 4), "depsign": (DS, 3)}
out = {}


def preds(cond, model, ds, cuts="fitted"):
    k = CORPORA[ds][1]
    labels = CORPORA[ds][0].LABELS
    if cond in ("C3", "C4"):
        cc = None if cuts == "fitted" else (
            A.CLINICAL_CUTOFFS if k == 4 else A.CLINICAL_CUTOFFS_3)
        a = A.build(model, condition=cond, dataset=ds, cutoffs=cc)
        return {i: (t, p) for i, t, p in zip(a["ids"], a["y_true"], a["y_pred"])}, a["cutoffs"]
    recs = A.load_run(cond, model, "test", dataset=ds)
    ok = [r for r in recs if r.get("pred")]
    return {r["id"]: (labels.index(r["gold"]), labels.index(r["pred"])) for r in ok}, None


def ev(d, k):
    y = np.array([v[0] for v in d.values()]); p = np.array([v[1] for v in d.values()])
    return E.evaluate(y, p, k=k)


def delta(d1, d2, k, metric="qwk"):
    ids = sorted(set(d1) & set(d2))
    y = np.array([d1[i][0] for i in ids])
    return E.paired_delta(y, np.array([d1[i][1] for i in ids]),
                          np.array([d2[i][1] for i in ids]), metric, k=k)


for ds, (mod, k) in CORPORA.items():
    out[ds] = {"k": k, "cells": {}, "deltas": {}, "density": {}}
    df = mod.load(); te = df[df.split == "test"]
    out[ds]["n_test"] = len(te)
    out[ds]["dist"] = {l: int((te.label == l).sum()) for l in mod.LABELS}
    out[ds]["floor"] = ev({i: (y, np.bincount(te.y, minlength=k).argmax())
                           for i, y in enumerate(te.y)}, k)
    for model in MODELS:
        n = SHORT[model]
        store = {}
        for cond in ("C1", "C2", "C3", "C4"):
            for cuts in (("fitted", "a priori") if cond in ("C3", "C4") else ("fitted",)):
                try:
                    d, cc = preds(cond, model, ds, cuts)
                except SystemExit:
                    continue
                tag = cond if cond in ("C1", "C2") else f"{cond} {cuts}"
                store[tag] = d
                m = ev(d, k)
                out[ds]["cells"][f"{n}|{tag}"] = {
                    "qwk": m["qwk"], "ci": m["qwk_ci"], "mae": m["mae"], "acc": m["acc"],
                    "macro_f1": m["macro_f1"], "sev_missed": m["severe_missed_n"],
                    "n_sev": m["n_gold_severe"], "over": m["over_rate"],
                    "under": m["under_rate"], "cutoffs": cc}
        for a, b in (("C1", "C2"), ("C2", "C3 fitted"), ("C2", "C3 a priori"),
                     ("C2", "C4 fitted"), ("C2", "C4 a priori")):
            if a in store and b in store:
                dd = delta(store[a], store[b], k)
                out[ds]["deltas"][f"{n}|{b} - {a}"] = {
                    "d": dd["delta"], "ci": dd["ci"], "sig": dd["excludes_zero"]}
        for cond, inv in (("C3", C.ITEMS), ("C4", C.BDI_ITEMS)):
            recs = [r for r in A.load_run(cond, model, "test", dataset=ds) if r.get("items")]
            if not recs:
                continue
            p = [sum(1 for v in r["items"].values() if v["status"] == "present") for r in recs]
            ab = sum(1 for r in recs for v in r["items"].values() if v["status"] == "absent")
            g = [v for r in recs for v in (r.get("grounding") or {}).values() if v is not None]
            out[ds]["density"][f"{n}|{cond}"] = {
                "median": int(np.median(p)), "mean": float(np.mean(p)), "max": int(max(p)),
                "n_items": len(inv), "absent_pct": ab / (len(recs) * len(inv)) * 100,
                "grounded_pct": sum(g) / len(g) * 100 if g else None}

out["baselines"] = json.load(open("results/phase2_baseline.json"))
json.dump(out, open("paper/numbers.json", "w"), indent=1, default=float)

# console summary
for ds in CORPORA:
    o = out[ds]
    print(f"\n===== {ds}  n_test={o['n_test']}  {o['dist']}  floor acc={o['floor']['acc']:.3f} "
          f"qwk={o['floor']['qwk']:.3f} mae={o['floor']['mae']:.3f} =====")
    for kk, v in o["cells"].items():
        cc = "" if not v["cutoffs"] else "[" + ",".join(f"{float(c):g}" for c in v["cutoffs"]) + "]"
        print(f"  {kk:<38} QWK {v['qwk']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] "
              f"MAE {v['mae']:.3f} acc {v['acc']:.3f} sev {v['sev_missed']}/{v['n_sev']} {cc}")
    for kk, v in o["deltas"].items():
        print(f"  D {kk:<36} {v['d']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] "
              f"{'SIG' if v['sig'] else 'ns'}")
