"""Metric suite shared by the baseline (Phase 2) and the LLM conditions (Phase 4)."""
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score, confusion_matrix, f1_score, accuracy_score,
)

LABELS = ["minimum", "mild", "moderate", "severe"]
SEVERE = 3


def _core(yt: np.ndarray, yp: np.ndarray, k: int = 4) -> dict:
    return {
        "qwk": cohen_kappa_score(yt, yp, weights="quadratic", labels=range(k)),
        "mae": float(np.abs(yt - yp).mean()),
        "acc": accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, average="macro", labels=range(k), zero_division=0),
    }


def evaluate(y_true, y_pred, n_boot: int = 2000, seed: int = 0, k: int = 4) -> dict:
    """Full metric suite with bootstrap CIs.

    Bootstrap over the test set, not seed-replication: at temperature 0 seed
    replication estimates nothing, and the question a reviewer actually asks is
    whether a gap exceeds test-set sampling noise.
    """
    yt, yp = np.asarray(y_true, int), np.asarray(y_pred, int)
    top = k - 1                      # index of the most severe class
    m = _core(yt, yp, k)

    # Error direction. Under-estimation is the clinically dangerous direction.
    err = yp - yt
    wrong = err != 0
    m["under_rate"] = float((err < 0).mean())
    m["over_rate"] = float((err > 0).mean())
    m["under_share_of_errors"] = float((err < 0).sum() / wrong.sum()) if wrong.any() else 0.0

    # Headline safety metric: gold-Severe posts predicted below Severe.
    sev = yt == top
    m["n_gold_severe"] = int(sev.sum())
    m["severe_missed_rate"] = float((yp[sev] < top).mean()) if sev.any() else float("nan")
    m["severe_missed_n"] = int((yp[sev] < top).sum())
    m["severe_mean_shortfall"] = float((top - yp[sev]).mean()) if sev.any() else float("nan")

    m["confusion"] = confusion_matrix(yt, yp, labels=range(k)).tolist()
    m["per_class_f1"] = f1_score(yt, yp, average=None, labels=range(k), zero_division=0).tolist()

    rng = np.random.default_rng(seed)
    boot = {k: [] for k in ("qwk", "mae", "acc", "macro_f1", "severe_missed_rate")}
    n = len(yt)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        b = _core(yt[i], yp[i], k)
        s = yt[i] == top
        b["severe_missed_rate"] = (yp[i][s] < top).mean() if s.any() else np.nan
        for name in boot:
            boot[name].append(b[name])
    for name, v in boot.items():
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        m[f"{name}_ci"] = [float(lo), float(hi)]
    return m


def majority_floor(y_true, k: int = 4) -> dict:
    yt = np.asarray(y_true, int)
    maj = np.bincount(yt, minlength=k).argmax()
    return evaluate(yt, np.full_like(yt, maj), k=k)


def fmt(name: str, m: dict) -> str:
    return (
        f"{name:<28} "
        f"QWK {m['qwk']:.3f} [{m['qwk_ci'][0]:.3f},{m['qwk_ci'][1]:.3f}]  "
        f"MAE {m['mae']:.3f}  Acc {m['acc']:.3f}  MacroF1 {m['macro_f1']:.3f}  "
        f"SevMissed {m['severe_missed_rate']:.3f} "
        f"[{m['severe_missed_rate_ci'][0]:.3f},{m['severe_missed_rate_ci'][1]:.3f}] "
        f"({m['severe_missed_n']}/{m['n_gold_severe']})"
    )


def paired_delta(y_true, pred_a, pred_b, metric: str = "qwk",
                 n_boot: int = 4000, seed: int = 0, k: int = 4) -> dict:
    """Bootstrap CI for metric(b) - metric(a) on the SAME posts.

    Conditions are evaluated on identical test items, so the comparison is paired.
    Overlapping marginal CIs do not imply the difference is indistinguishable from
    zero; this resamples posts once per replicate and scores both conditions on
    that same resample, which is the test the comparison actually calls for.
    """
    yt = np.asarray(y_true, int)
    a, b = np.asarray(pred_a, int), np.asarray(pred_b, int)

    top = k - 1

    def f(y, p):
        if metric == "qwk":
            return cohen_kappa_score(y, p, weights="quadratic", labels=range(k))
        if metric == "mae":
            return float(np.abs(y - p).mean())
        if metric == "severe_missed":
            s = y == top
            return float((p[s] < top).mean()) if s.any() else np.nan
        raise ValueError(metric)

    obs = f(yt, b) - f(yt, a)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(yt), len(yt))
        d.append(f(yt[i], b[i]) - f(yt[i], a[i]))
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return {"metric": metric, "delta": float(obs), "ci": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "p_two_sided": float(2 * min((np.array(d) <= 0).mean(),
                                         (np.array(d) >= 0).mean()))}
