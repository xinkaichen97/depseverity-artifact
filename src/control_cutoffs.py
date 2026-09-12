"""Control: does C3 still beat C2 when its cutoffs never see a training label?"""
import numpy as np
import aggregate as A
import conditions as C
import evaluate as E
import results as R

out = ["# Control — fitted vs a priori cutoffs\n",
       "C3's cutoffs are three parameters fitted on training labels; C1 and C2 get no",
       "such affordance. If C3's advantage survives replacing them with an a priori",
       f"clinical rule (DSM-5 >=5 of 9 symptoms -> severe; cutoffs {list(A.CLINICAL_CUTOFFS)}),",
       "the advantage is not an artefact of that asymmetry.\n",
       "| Model | C3 variant | cutoffs | QWK [95% CI] | MAE | Acc | Severe missed |",
       "|---|---|---|---|---:|---:|---|"]
rows = {}
for model in R.MODELS:
    name = R.SHORT[model]
    for tag, cuts in (("fitted", None), ("a priori", A.CLINICAL_CUTOFFS)):
        try:
            agg = A.build(model, cutoffs=cuts)
        except SystemExit as e:
            out.append(f"| {name} | {tag} | — | _{e}_ | | | |")
            continue
        y, p = np.array(agg["y_true"]), np.array(agg["y_pred"])
        m = E.evaluate(y, p)
        rows[(name, tag)] = (y, p, m)
        cs = ", ".join(f"{float(c):g}" for c in agg["cutoffs"])
        out.append(f"| {name} | {tag} | [{cs}] | **{m['qwk']:.3f}** "
                   f"[{m['qwk_ci'][0]:.3f}, {m['qwk_ci'][1]:.3f}] | {m['mae']:.3f} | "
                   f"{m['acc']:.3f} | {m['severe_missed_rate']:.3f} "
                   f"({m['severe_missed_n']}/{m['n_gold_severe']}) |")

out += ["", "## Does the C3 advantage over C2 survive? (paired, 4000 resamples)\n",
        "| Model | Comparison | dQWK [95% CI] | sig |", "|---|---|---|:--:|"]
for model in R.MODELS:
    name = R.SHORT[model]
    got = R.label_preds("C2", model)
    if not got or len(got[0]) < 700:
        continue
    y2, p2 = got[0], got[1]
    for tag in ("fitted", "a priori"):
        if (name, tag) not in rows:
            continue
        _, p3, _ = rows[(name, tag)]
        d = E.paired_delta(y2, p2, p3, "qwk")
        out.append(f"| {name} | C3({tag}) - C2 | {d['delta']:+.3f} "
                   f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | "
                   f"{'**yes**' if d['excludes_zero'] else 'no'} |")

txt = "\n".join(out)
open("results/control_cutoffs.md", "w").write(txt)
print(txt)
