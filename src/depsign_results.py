"""Second-corpus results: the pre-registered DepSign replication."""
import json
import numpy as np

import aggregate as A
import data_depsign as DS
import evaluate as E

MODELS = ["ollama:qwen3.5:9b", "deepseek:deepseek-flash", "anthropic:claude-sonnet-5"]
SHORT = {"ollama:qwen3.5:9b": "qwen3.5:9b", "deepseek:deepseek-flash": "V4.1-Flash",
         "anthropic:claude-sonnet-5": "sonnet-5"}
K = len(DS.LABELS)


def label_preds(cond, model):
    recs = A.load_run(cond, model, "test", dataset="depsign")
    ok = [r for r in recs if r.get("pred")]
    return {r["id"]: (DS.LABELS.index(r["gold"]), DS.LABELS.index(r["pred"])) for r in ok}


def c3_preds(model, cutoffs):
    a = A.build(model, condition="C3", dataset="depsign", cutoffs=cutoffs)
    return ({i: (t, p) for i, t, p in zip(a["ids"], a["y_true"], a["y_pred"])},
            a["cutoffs"])


def aligned(d1, d2):
    """Paired comparisons need identical posts; a parse failure drops one."""
    ids = sorted(set(d1) & set(d2))
    y = np.array([d1[i][0] for i in ids])
    return y, np.array([d1[i][1] for i in ids]), np.array([d2[i][1] for i in ids])


out = ["# Second corpus — DepSign (pre-registered replication)\n",
       "Identical prompts, harness, cutoff protocol and models; only the corpus changes.",
       "706-post stratified subsample of DepSign's **official test split** (50 severe);",
       "cutoffs fitted on 600 posts from its **official train split**. Three classes, so",
       "two cutoffs and the a priori rule is DSM-5 >=5 of 9 -> severe, 0 -> not depression.\n",
       "| Model | Condition | cutoffs | QWK [95% CI] | MAE | Acc | Top-class missed |",
       "|---|---|---|---|---:|---:|---|"]
store = {}
for model in MODELS:
    n = SHORT[model]
    for cond, nice in (("C1", "C1 (direct)"), ("C2", "C2 (CoT)")):
        d = label_preds(cond, model)
        store[(n, cond)] = d
        y = np.array([v[0] for v in d.values()]); p = np.array([v[1] for v in d.values()])
        m = E.evaluate(y, p, k=K)
        out.append(f"| {n} | {nice} | — | **{m['qwk']:.3f}** [{m['qwk_ci'][0]:.3f}, "
                   f"{m['qwk_ci'][1]:.3f}] | {m['mae']:.3f} | {m['acc']:.3f} | "
                   f"{m['severe_missed_n']}/{m['n_gold_severe']} |")
    for tag, cuts in (("C3 fitted", None), ("C3 a priori", A.CLINICAL_CUTOFFS_3)):
        d, cc = c3_preds(model, cuts)
        store[(n, tag)] = d
        y = np.array([v[0] for v in d.values()]); p = np.array([v[1] for v in d.values()])
        m = E.evaluate(y, p, k=K)
        cs = "[" + ", ".join(f"{float(c):g}" for c in cc) + "]"
        out.append(f"| {n} | {tag} | {cs} | **{m['qwk']:.3f}** [{m['qwk_ci'][0]:.3f}, "
                   f"{m['qwk_ci'][1]:.3f}] | {m['mae']:.3f} | {m['acc']:.3f} | "
                   f"{m['severe_missed_n']}/{m['n_gold_severe']} |")

out += ["", "## Primary pre-registered comparison (paired, 4000 resamples)\n",
        "| Model | Comparison | dQWK [95% CI] | sig | dMAE |", "|---|---|---|:--:|---|"]
for model in MODELS:
    n = SHORT[model]
    for a_tag, tag in (("C1", "C2"), ("C2", "C3 fitted"), ("C2", "C3 a priori")):
        y, pa, pb = aligned(store[(n, a_tag)], store[(n, tag)])
        d = E.paired_delta(y, pa, pb, "qwk", k=K)
        dm = E.paired_delta(y, pa, pb, "mae", k=K)
        out.append(f"| {n} | {tag} − {a_tag} | {d['delta']:+.3f} [{d['ci'][0]:+.3f}, "
                   f"{d['ci'][1]:+.3f}] | {'**yes**' if d['excludes_zero'] else 'no'} | "
                   f"{dm['delta']:+.3f} [{dm['ci'][0]:+.3f}, {dm['ci'][1]:+.3f}] "
                   f"{'sig' if dm['excludes_zero'] else 'ns'} |")

txt = "\n".join(out)
open("results/depsign_results.md", "w").write(txt)
print(txt)
