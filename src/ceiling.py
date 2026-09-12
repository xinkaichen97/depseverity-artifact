"""How much severity information is in the nine PHQ-9 items at all?

The count-plus-cutoffs rule is one aggregation function. If a *learned* function
over the same nine items does no better, then no cleverer aggregation of PHQ-9
will help either, and the limit is the inventory rather than the arithmetic.

This is deliberately an upper bound, not a fair comparison: the learned model has
far more free parameters than three cutoffs and far more than C1/C2 get (zero).
It answers "is the information there?", not "is C3 better than C2?".
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

import aggregate as A
import conditions as C
import evaluate as E
import results as R

VAL = {"present": 1.0, "unclear": 0.5, "absent": 0.0}


def feats(recs):
    X = np.array([[VAL[r["items"][k]["status"]] for k, _ in C.ITEMS] for r in recs])
    y = np.array([C.LABELS.index(r["gold"]) for r in recs])
    return X, y


def ordinal_fit_predict(Xtr, ytr, Xte, balanced: bool = True):
    """Cumulative binary decomposition, same family as the Phase 2 baseline."""
    cum = np.column_stack([
        LogisticRegression(max_iter=2000,
                           class_weight="balanced" if balanced else None)
        .fit(Xtr, (ytr > k).astype(int)).predict_proba(Xte)[:, 1] for k in range(3)])
    cum = np.minimum.accumulate(cum, axis=1)
    p = np.column_stack([1 - cum[:, 0], cum[:, 0] - cum[:, 1],
                         cum[:, 1] - cum[:, 2], cum[:, 2]])
    return p.argmax(1)


out = ["# Ceiling — how much is in the nine items?\n",
       "`count + fitted cutoffs` is the C3 rule (3 parameters). `learned 9-feature`",
       "fits an ordinal model over the same nine extracted statuses on the same fit",
       "split. It is an **upper bound** with many more parameters, not a fair rival to",
       "C2 — it bounds what any aggregation of PHQ-9 items could achieve.\n",
       "| Model | aggregation | QWK [95% CI] | MAE | Severe missed |",
       "|---|---|---|---:|---|"]
for model in R.MODELS:
    name = R.SHORT[model]
    try:
        agg = A.build(model)
    except SystemExit as e:
        out.append(f"| {name} | — | _{e}_ | | |")
        continue
    fit_recs = [r for r in A.load_run("C3", model, "fit") if r.get("items")]
    test_recs = [r for r in A.load_run("C3", model, "test") if r.get("items")]
    Xtr, ytr = feats(fit_recs)
    Xte, yte = feats(test_recs)

    m_cut = E.evaluate(np.array(agg["y_true"]), np.array(agg["y_pred"]))
    p_unb = ordinal_fit_predict(Xtr, ytr, Xte, balanced=False)
    m_unb = E.evaluate(yte, p_unb)
    m_lrn = E.evaluate(yte, ordinal_fit_predict(Xtr, ytr, Xte))
    for tag, m in (("count + fitted cutoffs (C3)", m_cut),
                   ("learned 9-feature, unweighted", m_unb),
                   ("learned 9-feature, class-balanced", m_lrn)):
        out.append(f"| {name} | {tag} | **{m['qwk']:.3f}** "
                   f"[{m['qwk_ci'][0]:.3f}, {m['qwk_ci'][1]:.3f}] | {m['mae']:.3f} | "
                   f"{m['severe_missed_rate']:.3f} "
                   f"({m['severe_missed_n']}/{m['n_gold_severe']}) |")
    d = E.paired_delta(yte, np.array(agg["y_pred"]), p_unb, "qwk")
    out.append(f"| {name} | _unweighted learned - C3_ | {d['delta']:+.3f} "
               f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] "
               f"{'(sig)' if d['excludes_zero'] else '**(ns - no headroom)**'} | | |")

# What does C2 get, for reference?
out += ["", "## Reference: C2 on the same test posts\n", "| Model | QWK |", "|---|---|"]
for model in R.MODELS:
    got = R.label_preds("C2", model)
    if got and len(got[0]) >= 700:
        m = E.evaluate(got[0], got[1], n_boot=500)
        out.append(f"| {R.SHORT[model]} | {m['qwk']:.3f} |")

txt = "\n".join(out)
open("results/ceiling.md", "w").write(txt)
print(txt)
