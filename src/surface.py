"""Are DepSeverity's labels recoverable from features unrelated to depressive criteria?

The human-annotator probe shows a criteria-following reader cannot reconstruct these
labels while models can. This tests the mechanistic account: that the labels track
surface properties of the posts rather than clinical content. Every feature here comes
from Dreaddit's own metadata, not from reading the post for symptoms.

No model calls. Fitted on the train split, evaluated on the frozen test split, using
the same ordinal construction and metrics as every other condition in the paper.
"""
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import aggregate as A
import data as D
import evaluate as E

DREADDIT = D.ROOT / "data/raw/dreaddit"


def load_features():
    dr = pd.concat([pd.read_csv(DREADDIT / f"dreaddit-{s}.csv") for s in ("train", "test")],
                   ignore_index=True)
    key = lambda s: s.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    dr = dr.assign(k=key(dr["text"])).drop_duplicates("k")
    df = D.load()
    df = df.assign(k=key(df["text"].astype(str)))
    m = df.merge(dr.drop(columns=["text"]), on="k", how="left", suffixes=("", "_dr"))
    m.index = df.index
    return m


def ordinal_fit(Xtr, ytr, Xte, k=4):
    cum = np.column_stack([
        LogisticRegression(max_iter=4000, class_weight="balanced")
        .fit(Xtr, (ytr > j).astype(int)).predict_proba(Xte)[:, 1] for j in range(k - 1)])
    cum = np.minimum.accumulate(cum, axis=1)
    p = np.column_stack([1 - cum[:, 0], cum[:, 0] - cum[:, 1],
                         cum[:, 1] - cum[:, 2], cum[:, 2]])
    return p.argmax(1)


LLM_CELLS = [
    ("DeepSeek C2 (CoT)",            "deepseek:deepseek-flash",   "C2", None),
    ("DeepSeek C3, fitted",          "deepseek:deepseek-flash",   "C3", "fit"),
    ("DeepSeek C3, a priori",        "deepseek:deepseek-flash",   "C3", "apriori"),
    ("Claude-Sonnet-5 C2 (CoT)",     "anthropic:claude-sonnet-5", "C2", None),
    ("Claude-Sonnet-5 C3, fitted",   "anthropic:claude-sonnet-5", "C3", "fit"),
    ("Claude-Sonnet-5 C3, a priori", "anthropic:claude-sonnet-5", "C3", "apriori"),
]


def llm_preds(model, cond, regime):
    """Test-split predictions keyed by post id, for a label-emitting or aggregated cell."""
    if cond in ("C1", "C2"):
        recs = [r for r in A.load_run(cond, model, "test") if "error" not in r and r.get("pred")]
        return {r["id"]: D.LABELS.index(r["pred"]) for r in recs}, \
               {r["id"]: D.LABELS.index(r["gold"]) for r in recs}
    agg = A.build(model, condition=cond,
                  cutoffs=None if regime == "fit" else A.CLINICAL_CUTOFFS)
    return dict(zip(agg["ids"], agg["y_pred"])), dict(zip(agg["ids"], agg["y_true"]))


def paired(ids, y_surface, p_surface, label="Dreaddit-feature model",
           desc="subreddit + stress + length + social + LIWC"):
    """Paired bootstrap of a Dreaddit-feature model against each LLM cell.

    The comparison the eyeball table invites is a paired one: both predictors score
    the same test posts, so marginal CIs are the wrong test.
    """
    sp = dict(zip(ids, p_surface))
    sy = dict(zip(ids, y_surface))
    out = ["", f"## Paired comparison: {label} vs each LLM cell", "",
           "Positive $\\Delta$ favors the LLM. Paired bootstrap on the posts both score.", "",
           f"| LLM cell | $\\kappa_w$ | $\\Delta$ vs {label} [95% CI] | $p$ |",
           "|---|---:|---|---:|"]
    print(f"\n{'LLM cell':<30}{'QWK':>7}   {'delta vs metadata':<26}{'p':>7}")
    for name, model, cond, regime in LLM_CELLS:
        pred, gold = llm_preds(model, cond, regime)
        both = [i for i in ids if i in pred]
        yt = np.array([sy[i] for i in both])
        assert (yt == np.array([gold[i] for i in both])).all(), f"gold mismatch: {name}"
        a = np.array([sp[i] for i in both])
        b = np.array([pred[i] for i in both])
        d = E.paired_delta(yt, a, b)
        q = E._core(yt, b)["qwk"]
        star = "*" if d["excludes_zero"] else ""
        print(f"{name:<30}{q:>7.3f}   {d['delta']:+.3f} [{d['ci'][0]:+.3f},"
              f"{d['ci'][1]:+.3f}]{star:<3}{d['p_two_sided']:>7.3f}  (n={len(both)})")
        out.append(f"| {name} | {q:.3f} | {d['delta']:+.3f} "
                   f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]{star} | {d['p_two_sided']:.3f} |")
    out += ["", f"`*` = 95% CI excludes zero. {label}: "
            f"$\\kappa_w$ {E._core(np.asarray(y_surface), np.asarray(p_surface))['qwk']:.3f} "
            f"({desc})."]
    return out


def main():
    m = load_features()
    tr, te = m[m.split == "train"], m[m.split == "test"]
    liwc = [c for c in m.columns if c.startswith(("lex_", "syntax_"))]
    social = ["social_karma", "social_upvote_ratio", "social_num_comments", "sentiment"]
    # Fields that do not read the post: `sentiment`, length and LIWC are computed from its text.
    nontext = ["social_karma", "social_upvote_ratio", "social_num_comments"]

    def build(cols_num, use_sub, use_stress, use_len):
        parts = []
        for frame in (tr, te):
            blocks = []
            if cols_num:
                blocks.append(frame[cols_num].astype(float).fillna(0.0).values)
            if use_sub:
                blocks.append(pd.get_dummies(m["subreddit"]).loc[frame.index].values.astype(float))
            if use_stress:
                blocks.append(frame[["stress", "stress_conf"]].astype(float).fillna(0).values)
            if use_len:
                w = frame["text"].astype(str).str.split().str.len().values.reshape(-1, 1)
                blocks.append(np.hstack([w, np.log1p(w)]))
            parts.append(np.hstack(blocks))
        sc = StandardScaler().fit(parts[0])
        return sc.transform(parts[0]), sc.transform(parts[1])

    ytr, yte = tr.y.values, te.y.values
    preds = {}
    rows = [
        ("source subreddit only (10 dummies)", ([], True, False, False)),
        ("post length only (2 features)", ([], False, False, True)),
        ("Dreaddit stress label + confidence", ([], False, True, False)),
        ("subreddit + stress + length", ([], True, True, True)),
        ("+ social signals (karma, votes, comments)", (social, True, True, True)),
        ("+ LIWC/syntax lexicon counts", (liwc + social, True, True, True)),
        ("non-text fields only: subreddit + stress + karma/votes/comments",
         (nontext, True, True, False)),
    ]
    out = ["# Are the labels recoverable from non-criteria features?\n",
           "Every feature comes with Dreaddit or is a surface property; none is a symptom",
           "annotation. Length, `sentiment` (in the social signals) and the LIWC/syntax counts",
           "are computed from the post text; the last row uses only fields that do not read",
           "it. Fitted on train, evaluated on the frozen test split, same ordinal construction",
           "and metrics as the LLM conditions.\n",
           "| Feature set | $\\kappa_w$ [95% CI] | MAE | Acc |", "|---|---|---:|---:|"]
    print(f"{'feature set':<44}{'QWK':>7}{'   95% CI':<18}{'MAE':>7}{'Acc':>7}")
    for name, (cols, sub, st, ln) in rows:
        Xtr, Xte = build(cols, sub, st, ln)
        yp = ordinal_fit(Xtr, ytr, Xte)
        preds[name] = yp
        mm = E.evaluate(yte, yp)
        print(f"{name:<44}{mm['qwk']:>7.3f}   [{mm['qwk_ci'][0]:.3f},{mm['qwk_ci'][1]:.3f}]"
              f"{mm['mae']:>9.3f}{mm['acc']:>7.3f}")
        out.append(f"| {name} | **{mm['qwk']:.3f}** [{mm['qwk_ci'][0]:.3f}, "
                   f"{mm['qwk_ci'][1]:.3f}] | {mm['mae']:.3f} | {mm['acc']:.3f} |")
    out += ["", "Reference points on the same test split: best LLM condition "
            "$\\kappa_w = 0.526$ (C3 fitted, DeepSeek-V4.1-Flash); chain-of-thought "
            "0.462--0.504; TF-IDF text baseline 0.374; majority class 0.000."]
    out += paired(te.index.values, yte, preds[rows[-2][0]])
    out += paired(te.index.values, yte, preds[rows[-1][0]], label="non-text model",
                  desc="subreddit + stress + karma/votes/comments; nothing computed from the text")
    open("results/surface_features.md", "w").write("\n".join(out))
    print()
    print("reference: best LLM 0.526 | CoT 0.462-0.504 | TF-IDF 0.374 | majority 0.000")


if __name__ == "__main__":
    main()
