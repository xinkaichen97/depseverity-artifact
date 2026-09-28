"""C4 under BDI-II's own severity bands, rescaled onto the 0-21 present count.

paper_numbers.py reports "C4 a priori" with the DSM-5 cutoffs carried over from C3
(A.CLINICAL_CUTOFFS). The second a priori rule for C4 is BDI-II's published bands
(A.BDI_CUTOFFS_*, derived in aggregate.py). This script produces that comparison.
"""
import numpy as np
import aggregate as A
import data as D
import data_depsign as DS
import evaluate as E

MODELS = ["ollama:qwen3.5:9b", "deepseek:deepseek-flash", "anthropic:claude-sonnet-5"]
SHORT = {"ollama:qwen3.5:9b": "Qwen3.5-9B", "deepseek:deepseek-flash": "DeepSeek-V4.1-Flash",
         "anthropic:claude-sonnet-5": "Claude-Sonnet-5"}
CORPORA = {"depseverity": (D, 4, A.BDI_CUTOFFS_4),
           "depsign": (DS, 3, A.BDI_CUTOFFS_3)}

out = ["# C4 under BDI-II's own bands\n",
       "BDI-II's bands are defined on its native 0-63 sum of graded items. Rescaled onto",
       "our 0-21 binary present count (aggregate.py), they give the cutoffs below. C4 is",
       "compared with C2 on the same posts.\n",
       "| Corpus | Model | cutoffs | C4 QWK | dQWK vs C2 [95% CI] | sig |",
       "|---|---|---|---:|---|:--:|"]

for ds, (mod, k, cuts) in CORPORA.items():
    L = mod.LABELS
    for model in MODELS:
        name = SHORT[model]
        c2 = {r["id"]: (L.index(r["gold"]), L.index(r["pred"]))
              for r in A.load_run("C2", model, "test", dataset=ds) if r.get("pred")}
        try:
            agg = A.build(model, condition="C4", dataset=ds, cutoffs=cuts)
        except SystemExit as e:
            out.append(f"| {ds} | {name} | — | _{e}_ | | |")
            continue
        c4 = dict(zip(agg["ids"], zip(agg["y_true"], agg["y_pred"])))
        ids = sorted(set(c2) & set(c4))
        y = np.array([c2[i][0] for i in ids])
        p2 = np.array([c2[i][1] for i in ids])
        p4 = np.array([c4[i][1] for i in ids])
        m = E.evaluate(y, p4, k=k)
        d = E.paired_delta(y, p2, p4, "qwk", k=k)
        cs = ", ".join(f"{float(c):g}" for c in cuts)
        out.append(f"| {ds} | {name} | [{cs}] | {m['qwk']:.3f} | {d['delta']:+.3f} "
                   f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | "
                   f"{'**yes**' if d['excludes_zero'] else 'no'} |")

txt = "\n".join(out)
open("results/bdi_bands.md", "w").write(txt)
print(txt)
