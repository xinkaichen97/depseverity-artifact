"""C3 with stricter count thresholds: SEVERE requires seven present criteria.

Sec. IV-B ("Stricter thresholds do not help"). Our choice, not a DSM-5 rule: four bands
0-1 / 2-4 / 5-6 / 7+, i.e. cutoffs [1.5, 4.5, 6.5]. DepSign has three labels, so the four
bands collapse in two ways that keep SEVERE at seven or more: merging the two lowest bands
([4.5, 6.5]) or the two middle bands ([1.5, 6.5]). The standard a priori rule is listed for
reference. Every rule is compared with C2 on the same posts.
"""
import numpy as np
import aggregate as A
import data as D
import data_depsign as DS
import evaluate as E

MODELS = ["ollama:qwen3.5:9b", "deepseek:deepseek-flash", "anthropic:claude-sonnet-5"]
SHORT = {"ollama:qwen3.5:9b": "Qwen3.5-9B", "deepseek:deepseek-flash": "DeepSeek-V4.1-Flash",
         "anthropic:claude-sonnet-5": "Claude-Sonnet-5"}
RULES = {
    "depseverity": (D, 4, [("standard a priori", A.CLINICAL_CUTOFFS),
                           ("stricter", (1.5, 4.5, 6.5))]),
    "depsign": (DS, 3, [("standard a priori", A.CLINICAL_CUTOFFS_3),
                        ("stricter, lowest two merged", (4.5, 6.5)),
                        ("stricter, middle two merged", (1.5, 6.5))]),
}

out = ["# C3 with stricter count thresholds\n",
       "SEVERE requires seven present criteria (bands 0-1 / 2-4 / 5-6 / 7+). On DepSign the",
       "four bands collapse to three labels in two ways that keep that requirement. Each rule",
       "is compared with C2 on the same posts (paired bootstrap, 4000 resamples).\n",
       "| Corpus | Model | Rule | cutoffs | C3 QWK | SEVERE missed | dQWK vs C2 [95% CI] | sig |",
       "|---|---|---|---|---:|---|---|:--:|"]

for ds, (mod, k, rules) in RULES.items():
    L = mod.LABELS
    for model in MODELS:
        name = SHORT[model]
        c2 = {r["id"]: (L.index(r["gold"]), L.index(r["pred"]))
              for r in A.load_run("C2", model, "test", dataset=ds) if r.get("pred")}
        for rule, cuts in rules:
            agg = A.build(model, condition="C3", dataset=ds, cutoffs=cuts)
            c3 = dict(zip(agg["ids"], zip(agg["y_true"], agg["y_pred"])))
            ids = sorted(set(c2) & set(c3))
            y = np.array([c2[i][0] for i in ids])
            p2 = np.array([c2[i][1] for i in ids])
            p3 = np.array([c3[i][1] for i in ids])
            m = E.evaluate(y, p3, k=k)
            d = E.paired_delta(y, p2, p3, "qwk", k=k)
            cs = ", ".join(f"{float(c):g}" for c in cuts)
            out.append(f"| {ds} | {name} | {rule} | [{cs}] | {m['qwk']:.3f} | "
                       f"{m['severe_missed_n']}/{m['n_gold_severe']} | {d['delta']:+.3f} "
                       f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | "
                       f"{'**yes**' if d['excludes_zero'] else 'no'} |")

txt = "\n".join(out)
open("results/stricter_thresholds.md", "w").write(txt)
print(txt)
