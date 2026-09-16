"""Deterministic criteria -> severity.

The model never produces a severity label in C3. The label is assigned here, by
code, from the extracted PHQ-9 items.

Scoring note (empirical, see notes/findings.md): the spec's score is
`1*present + 0.5*unclear + 0*absent`. Across all three models `absent` was used
for 3 of 864 dev judgments, so `unclear = 9 - present` and the score reduces to
`0.5*present + 4.5` - an affine function of the present count, which any
monotone cutoff rule is invariant to. The 0.5 weighting is therefore inert on
this corpus. We score the raw present count and report the collapse, rather than
carrying a weighting that provably does nothing.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

import conditions as C

RUNS = Path(__file__).resolve().parents[1] / "runs"
N_ITEMS = len(C.ITEMS)


def score(items: dict, weight_unclear: float = 0.0, inventory=None) -> float:
    """Symptom score. weight_unclear=0 is the present count; 0.5 is the spec's rule."""
    s = 0.0
    for key, _ in (C.ITEMS if inventory is None else inventory):
        st = items[key]["status"]
        s += 1.0 if st == "present" else (weight_unclear if st == "unclear" else 0.0)
    return s


_STATUS = ("present", "absent", "unclear")


def recover_items(raw: str, inventory) -> dict | None:
    """Rescue the per-item statuses from a response whose JSON did not parse.

    Every parse failure we see is a malformed escape inside an `evidence` string
    (a censored expletive, a doubled quote), not a missing or partial extraction:
    the nine `status` fields are all there. The score counts statuses and never
    reads `evidence`, so recovering the statuses recovers the label exactly.

    Returns None unless every item in the inventory is found with a valid status,
    so a genuinely truncated response is still dropped rather than half-scored.
    """
    import re
    out = {}
    for key, _ in inventory:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*\{{', raw)
        if not m:
            return None
        st = re.search(r'"status"\s*:\s*"(\w+)"', raw[m.end():])
        if not st or st.group(1) not in _STATUS:
            return None
        out[key] = {"status": st.group(1), "evidence": ""}
    return out


def usable(recs: list[dict], inventory) -> list[dict]:
    """Records with items, recovering the statuses of any that failed to parse."""
    out = []
    for r in recs:
        if r.get("items"):
            out.append(r)
            continue
        got = recover_items(r.get("raw") or "", inventory)
        if got is not None:
            out.append({**r, "items": got, "recovered": True})
    return out


def load_run(condition: str, model: str, subset: str, seed: int = 0,
             dataset: str = "depseverity") -> list[dict]:
    import re
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", model)
    pre = "" if dataset == "depseverity" else f"{dataset}_"
    p = RUNS / f"{pre}{condition}_{safe}_{subset}_seed{seed}.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else []


def apply_cutoffs(scores: np.ndarray, cuts: tuple[float, float, float]) -> np.ndarray:
    return np.searchsorted(np.asarray(cuts), scores, side="right")


def fit_cutoffs(scores: np.ndarray, y: np.ndarray, n_items: int = N_ITEMS,
                k: int = 4) -> tuple[tuple, float]:
    """Exhaustive search over ordered cutoffs, maximising QWK on the fitting split.

    The score is a small integer, so the candidate grid is tiny and a full search
    beats any optimiser - no local optima, no seed dependence.
    """
    grid = np.arange(-0.5, n_items + 1.0, 0.5)
    best, best_q = None, -np.inf
    for cuts in itertools.combinations(grid, k - 1):
        q = cohen_kappa_score(y, apply_cutoffs(scores, cuts),
                              weights="quadratic", labels=range(k))
        if q > best_q:
            best, best_q = cuts, q
    return best, best_q


def collapse_stats(recs: list[dict], inventory=None) -> dict:
    """Evidence for the scoring note above: how often each status is actually used."""
    inv = C.ITEMS if inventory is None else inventory
    ok = [r for r in recs if r.get("items")]
    n = len(ok) * len(inv)
    c = {"present": 0, "absent": 0, "unclear": 0}
    for r in ok:
        for key, _ in inv:
            c[r["items"][key]["status"]] += 1
    return {"n_judgments": n, **c,
            "absent_rate": c["absent"] / n if n else float("nan")}


# A priori clinical cutoffs, chosen without reference to any label in this corpus.
# DSM-5 requires >=5 of the 9 criterion-A symptoms for a major depressive episode,
# so >=5 present -> severe; 0 present -> minimum; the interior splits 1-2 / 3-4.
# This is the control for "C3 only wins because its cutoffs saw training labels."
CLINICAL_CUTOFFS = (0.5, 2.5, 4.5)
# Three-class corpora (DepSign): 0 symptoms -> not depression, 1-4 -> moderate,
# >=5 -> severe, on the same DSM-5 anchor. Chosen without reference to any label.
CLINICAL_CUTOFFS_3 = (0.5, 4.5)

# BDI-II (21 items). The DSM-5 ">=5 of 9" anchor does not transfer to a 21-item
# inventory, so C4 is anchored on BDI-II's own published severity bands instead:
# 0-13 minimal / 14-19 mild / 20-28 moderate / 29-63 severe on its native 0-63 scale.
# Band upper bounds as fractions of the maximum (13/63, 19/63, 28/63) rescaled onto a
# 0-21 present-count give 4.33, 6.33, 9.33 -> cutoffs below. The rescaling assumes a
# binary present-count is proportional to a 0-3 severity sum, which is an approximation
# we state rather than hide. Chosen without reference to any label in either corpus.
BDI_CUTOFFS_4 = (4.5, 6.5, 9.5)
BDI_CUTOFFS_3 = (4.5, 9.5)      # collapse mild+moderate for the 3-class corpus


def build(model: str, seed: int = 0, cutoffs=None, condition: str = "C3",
          dataset: str = "depseverity") -> dict:
    """Fit cutoffs on the fit split, freeze, apply to test.

    Pass `cutoffs` to bypass fitting entirely and use an a priori rule instead.
    """
    import data as D
    import data_depsign as DS
    mod = D if dataset == "depseverity" else DS
    L = mod.LABELS
    k = len(L)
    inv = C.ITEMS_FOR[condition]
    n_items = len(inv)
    raw_test = load_run(condition, model, "test", seed, dataset)
    test = usable(raw_test, inv)
    if not test:
        raise SystemExit(f"missing {condition} test run for {model}")
    # A partial run silently yields a plausible-looking but wrong row (fewer gold-Severe
    # posts, different class balance). Refuse rather than report it.
    n_expected = int((mod.load()["split"] == "test").sum())
    if len(test) < n_expected:
        raise SystemExit(
            f"{condition} test run for {model} is INCOMPLETE: {len(test)}/{n_expected}")

    if cutoffs is None:
        fit = usable(load_run(condition, model, "fit", seed, dataset), inv)
        if not fit:
            raise SystemExit(f"missing {condition} fit run for {model}")
        sf = np.array([score(r["items"], inventory=inv) for r in fit])
        yf = np.array([L.index(r["gold"]) for r in fit])
        cuts, q_fit = fit_cutoffs(sf, yf, n_items, k)
    else:
        fit, cuts, q_fit = [], tuple(cutoffs), float("nan")

    st = np.array([score(r["items"], inventory=inv) for r in test])
    yt = np.array([L.index(r["gold"]) for r in test])
    pred = apply_cutoffs(st, cuts)
    return {"model": model, "condition": condition, "dataset": dataset,
            "k": k, "cutoffs": list(cuts), "qwk_fit": float(q_fit),
            "fitted": cutoffs is None, "n_fit": len(fit), "n_test": len(test),
            "ids": [r["id"] for r in test], "y_true": yt.tolist(),
            "y_pred": pred.tolist(), "scores": st.tolist(),
            "collapse_test": collapse_stats(test, inv)}


if __name__ == "__main__":
    import sys
    out = build(sys.argv[1] if len(sys.argv) > 1 else "deepseek:deepseek-v4-pro")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("ids", "y_true", "y_pred", "scores")}, indent=2))
