"""Symmetry control: give C1 and C2 the same labeled budget C3 gets.

C3 converts a symptom count into a label using thresholds fitted on the fitting split.
C1 and C2 emit a label directly and receive no labeled data at all, so the comparison
is confounded: C3's advantage could be the supervision rather than the structure.

Stripping C3's supervision (the a priori control) tests one direction. This tests the
other: fit a monotone relabeling of C1/C2's predicted class on the same fitting split,
freeze it, apply to test. A monotone map on k ordered classes has C(2k-1, k) forms --
35 for k=4, 10 for k=3 -- which is *less* flexible than the 3-of-21 threshold grid C3
searches, so this is a conservative version of the control.
"""
import itertools
import json

import numpy as np

import aggregate as A
import data as D
import data_depsign as DS
import evaluate as E

CORP = {"depseverity": (D, 4), "depsign": (DS, 3)}


def monotone_maps(k):
    """Every non-decreasing f: {0..k-1} -> {0..k-1}."""
    return [m for m in itertools.product(range(k), repeat=k)
            if all(m[i] <= m[i + 1] for i in range(k - 1))]


def load(cond, model, subset, ds):
    mod, _ = CORP[ds]
    L = mod.LABELS
    return {r["id"]: (L.index(r["gold"]), L.index(r["pred"]))
            for r in A.load_run(cond, model, subset, dataset=ds) if r.get("pred")}


def fit_map(fit, k):
    y = np.array([v[0] for v in fit.values()])
    p = np.array([v[1] for v in fit.values()])
    best, bq = None, -np.inf
    for m in monotone_maps(k):
        q = E._core(y, np.array([m[x] for x in p]), k)["qwk"]
        if q > bq:
            best, bq = m, q
    return best, bq


def main():
    out = ["# Symmetry control --- C1/C2 with the same labeled budget as C3\n",
           "A monotone relabeling of the predicted class, fitted on the same 600-post",
           "fitting split used for C3's thresholds, then frozen and applied to test.",
           "Fewer free parameters than C3's threshold search.\n",
           "| Corpus | Model | Cond. | raw $\\kappa_w$ | +cal $\\kappa_w$ | map | C3 fitted |",
           "|---|---|---|---:|---:|---|---:|"]
    deltas = []
    for ds, (mod, k) in CORP.items():
        models = ["ollama:qwen3.5:9b", "deepseek:deepseek-flash", "anthropic:claude-sonnet-5"]
        for model in models:
            short = model.split(":")[1][:12]
            try:
                c3 = A.build(model, condition="C3", dataset=ds)
                q3 = E._core(np.array(c3["y_true"]), np.array(c3["y_pred"]), k)["qwk"]
            except SystemExit:
                q3 = float("nan")
            for cond in ("C1", "C2"):
                fit = load(cond, model, "fit", ds)
                test = load(cond, model, "test", ds)
                if not fit or not test:
                    out.append(f"| {ds} | {short} | {cond} | _no fit run_ | | | |")
                    continue
                m, _ = fit_map(fit, k)
                y = np.array([v[0] for v in test.values()])
                p = np.array([v[1] for v in test.values()])
                pc = np.array([m[x] for x in p])
                qr = E._core(y, p, k)["qwk"]; qc = E._core(y, pc, k)["qwk"]
                out.append(f"| {ds} | {short} | {cond} | {qr:.3f} | **{qc:.3f}** | "
                           f"`{''.join(str(x) for x in m)}` | {q3:.3f} |")
                deltas.append((ds, short, cond, k, y, pc, p, q3, qc))

    out += ["", "## Does C3 still beat a supervision-matched C1/C2? (paired, 4000 resamples)\n",
            "| Corpus | Model | Comparison | $\\Delta\\kappa_w$ [95\\% CI] | sig |",
            "|---|---|---|---|:--:|"]
    for ds, (mod, k) in CORP.items():
        models = ["ollama:qwen3.5:9b", "deepseek:deepseek-flash", "anthropic:claude-sonnet-5"]
        for model in models:
            short = model.split(":")[1][:12]
            for cond in ("C2", "C1"):
                rows = [d for d in deltas if d[0] == ds and d[1] == short and d[2] == cond]
                if not rows:
                    continue
                _, _, _, k, y, pc, p, q3, qc = rows[0]
                try:
                    c3 = A.build(model, condition="C3", dataset=ds)
                    d3 = {i: q for i, q in zip(c3["ids"], c3["y_pred"])}
                except SystemExit:
                    continue
                test = load(cond, model, "test", ds)
                ids = sorted(set(test) & set(d3))
                yy = np.array([test[i][0] for i in ids])
                cal = np.array([monotone_maps(k)[0][0] for _ in ids])  # placeholder, replaced below
                m, _ = fit_map(load(cond, model, "fit", ds), k)
                cal = np.array([m[test[i][1]] for i in ids])
                c3p = np.array([d3[i] for i in ids])
                dd = E.paired_delta(yy, cal, c3p, "qwk", k=k)
                out.append(f"| {ds} | {short} | C3 fitted $-$ {cond}+cal | "
                           f"${dd['delta']:+.3f}$ [{dd['ci'][0]:+.3f}, {dd['ci'][1]:+.3f}] | "
                           f"{'**yes**' if dd['excludes_zero'] else 'no'} |")
    txt = "\n".join(out)
    open("results/calibration_control.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
