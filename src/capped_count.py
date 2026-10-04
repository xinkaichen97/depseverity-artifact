"""Equal-supervision control at equal resolution: cap C3's count at k levels.

Sec. V-C. C3's present count runs 0-9, while C2 emits one of k labels, so C3's fitted
thresholds have more room than C2's monotone relabeling (calibrate.py). Here C3's count
is capped at k-1 (levels 0, 1, ..., k-1+; for DepSeverity 0, 1, 2, 3+) and relabeled
with the same search over the same C(2k-1, k) monotone maps, fitted on the same 600-post
split, frozen and applied to test. If C3's lead over recalibrated C2 survives, it comes
from the extraction itself, not from C3's finer score.
"""
import numpy as np

import aggregate as A
import conditions as C
import data as D
import data_depsign as DS
import evaluate as E
from calibrate import fit_map, load

MODELS = ["ollama:qwen3.5:9b", "deepseek:deepseek-flash", "anthropic:claude-sonnet-5"]
SHORT = {"ollama:qwen3.5:9b": "Qwen3.5-9B", "deepseek:deepseek-flash": "DeepSeek-V4.1-Flash",
         "anthropic:claude-sonnet-5": "Claude-Sonnet-5"}
CORPORA = {"depseverity": (D, 4), "depsign": (DS, 3)}
INV = C.ITEMS_FOR["C3"]


def capped(model, subset, ds, mod, k):
    """id -> (gold, C3 present count capped at k-1)."""
    recs = A.usable(A.load_run("C3", model, subset, dataset=ds), INV)
    return {r["id"]: (mod.LABELS.index(r["gold"]),
                      min(int(A.score(r["items"], inventory=INV)), k - 1)) for r in recs}


def fmt(d):
    return (f"{d['delta']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | "
            f"{'**yes**' if d['excludes_zero'] else 'no'}")


out = ["# C3 capped to C2's resolution, against recalibrated C2\n",
       "C3's count is capped at k-1 and relabeled by the best monotone map on the fitting",
       "split, the same search C2+cal gets (calibrate.py). Paired bootstrap, 4000 resamples.\n",
       "| Corpus | Model | C3 capped map | C3 fitted $-$ C2+cal | sig | "
       "C3 capped $-$ C2+cal | sig |",
       "|---|---|---|---|:--:|---|:--:|"]

for ds, (mod, k) in CORPORA.items():
    for model in MODELS:
        c2_fit, c2_test = load("C2", model, "fit", ds), load("C2", model, "test", ds)
        if not c2_fit or not c2_test:
            continue
        m2, _ = fit_map(c2_fit, k)
        m3, _ = fit_map(capped(model, "fit", ds, mod, k), k)
        cap_test = capped(model, "test", ds, mod, k)
        full = A.build(model, condition="C3", dataset=ds)
        c3_fitted = dict(zip(full["ids"], full["y_pred"]))
        ids = sorted(set(c2_test) & set(cap_test) & set(c3_fitted))
        y = np.array([c2_test[i][0] for i in ids])
        c2cal = np.array([m2[c2_test[i][1]] for i in ids])
        d_fit = E.paired_delta(y, c2cal, np.array([c3_fitted[i] for i in ids]), "qwk", k=k)
        d_cap = E.paired_delta(y, c2cal, np.array([m3[cap_test[i][1]] for i in ids]), "qwk", k=k)
        out.append(f"| {ds} | {SHORT[model]} | `{''.join(map(str, m3))}` | "
                   f"{fmt(d_fit)} | {fmt(d_cap)} |")

txt = "\n".join(out)
open("results/capped_count.md", "w").write(txt)
print(txt)
