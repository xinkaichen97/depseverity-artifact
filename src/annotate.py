"""Label-noise ceiling: draw a blind annotation set, then score it against gold.

We cannot access either corpus's annotation documentation, so we do not know the
inter-annotator agreement our metrics are measured against. Two DepSeverity posts carry
conflicting gold labels on byte-identical text, so the floor is not trivial. This gives
a single-annotator estimate of it.

  python src/annotate.py make    -> writes notes/annotation_blind.csv  (no gold labels)
  python src/annotate.py score   -> reads it back and reports agreement with gold
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
import data_depsign as DS
import evaluate as E

ROOT = Path(__file__).resolve().parents[1]
CORP = {"depseverity": (D, 25), "depsign": (DS, 34)}   # ~100 posts per corpus
SEED = 20260912


def paths(ds):
    return (ROOT / f"notes/annotation_blind_{ds}.csv",
            ROOT / f"notes/annotation_rubric_{ds}.md")

RUBRIC_DEPSIGN = """# Blind annotation task — DepSign (3 levels)

Assign one label per post from: not depression, moderate, severe.

This corpus's annotation guidelines are not public --- the shared-task findings paper
defers to a dataset paper we could not obtain --- so the rubric below is ours, not
theirs. Agreement therefore measures how reproducible the labels are from the text
under a reasonable reading, not fidelity to the original annotators' intent.

  not depression  No or minimal indication of depressive symptoms. The writer may be
                  distressed, asking for advice, or discussing someone else.
  moderate        Depressive symptoms clearly present --- low mood, anhedonia,
                  hopelessness, sleep or energy disturbance --- without the markers
                  below.
  severe          Pervasive symptoms plus at least one intensity marker: active
                  suicidal ideation (method, plan, intent, timeframe, or recent
                  attempt), hopelessness about the future, or functional collapse.

Passive ideation ("I wish I wasn't here") is one symptom among others and does not by
itself make a post severe; active ideation does. Record which in the `ideation` column
(`passive` / `active` / blank) so the boundary can be varied afterwards.

Rules:
  1. Judge only what the post states or clearly implies.
  2. Do NOT look at gold labels, model outputs, or any results file while annotating.
  3. Fill the `my_label` column only.
  4. If genuinely undecidable, write `unsure` rather than guessing.
"""

RUBRIC = """# Blind annotation task — depression severity

Assign one label per post from: minimum, mild, moderate, severe.

The corpus's own guidelines are not public. DepSeverity's four levels are BDI-II's
severity bands, and its symptom-level scheme is the DSM-5 criterion-A set, so use:

  minimum   No or minimal indication of depressive symptoms. The post may describe
            stress, conflict, or practical hardship without depressive content.
  mild      Some depressive symptoms evident, limited in number and intensity.
  moderate  Several depressive symptoms, clear distress, some suggestion of impact
            on functioning.
  severe    Symptoms pervasive across the post AND at least one intensity marker:
            active suicidal ideation, hopelessness about the future, or functional
            collapse (cannot work, cannot leave bed, cannot care for self).

## Suicidal ideation: the moderate/severe boundary

Ideation alone does not decide the band. On BDI-II, thoughts of death score 1 of a
possible 3 on one of twenty-one items, and the severe band begins at 29 of 63 — so
passive ideation contributes to the total without determining it.

  PASSIVE  "I wish I wasn't here", "it would be easier not to wake up", "I think
           about dying sometimes". Treat as one symptom among others and decide the
           band on the overall picture. With few other symptoms this is usually
           `moderate`.
  ACTIVE   A method, a plan, stated intent, a timeframe, or a recent attempt.
           Treat as `severe` regardless of the rest of the post.

Set the `ideation` column to `passive`, `active`, or leave blank when absent. This is
recorded so the boundary rule can be varied afterwards; it does not change what you
should put in `my_label`.

Rules:
  1. Judge only what the post states or clearly implies. Do not infer a history.
  2. Do NOT look at gold labels, model outputs, or any results file while annotating.
  3. Work in one sitting if you can; note roughly how long it took.
  4. Fill `my_label` and, where relevant, `ideation`. Leave everything else untouched.
  5. If genuinely undecidable, write `unsure` rather than guessing — those are
     reported separately rather than scored.
  6. Whatever you decide on a borderline case, apply the same rule for the rest. We
     are measuring whether these labels are reproducible from the text, so internal
     consistency matters more than matching an annotator you cannot consult.
"""


def make(ds="depseverity"):
    """Extend the annotation set to PER_CLASS per class, preserving existing work.

    Already-annotated rows are kept verbatim; only the shortfall is drawn, from posts
    not already sampled. Re-running is therefore safe and additive.
    """
    mod, per = CORP[ds]
    out_csv, out_rub = paths(ds)
    te = mod.load()
    te = te[te.split == "test"]

    existing = pd.read_csv(out_csv, index_col="id") if out_csv.exists() else None
    have = set(existing.index) if existing is not None else set()

    picks = []
    for lab in mod.LABELS:
        pool = te[(te.label == lab) & (~te.index.isin(have))]
        n_have = int(te.loc[list(have & set(te.index)), "label"].eq(lab).sum()) if have else 0
        need = max(0, min(per, int((te.label == lab).sum())) - n_have)
        if need:
            picks.append(pool.sample(min(need, len(pool)), random_state=SEED))
    new = pd.concat(picks) if picks else te.iloc[0:0]
    add = new[["text"]].copy()
    add.insert(0, "ideation", "")
    add.insert(0, "my_label", "")
    out = pd.concat([existing, add]) if existing is not None else add
    out_csv.parent.mkdir(exist_ok=True)
    out.to_csv(out_csv, index_label="id")
    print(f"  kept {len(have)} existing, added {len(add)} new")
    out_rub.write_text(RUBRIC if ds == "depseverity" else RUBRIC_DEPSIGN)
    print(f"wrote {out_csv.name} ({len(out)} posts, {per}/class, shuffled) "
          f"+ {out_rub.name}")
    print(f"  labels to use: {mod.LABELS}")


def score(ds="depseverity"):
    mod, _ = CORP[ds]
    out_csv, _ = paths(ds)
    k = len(mod.LABELS)
    ann = pd.read_csv(out_csv, index_col="id")
    gold = mod.load()["label"]
    ann["my_label"] = ann["my_label"].astype(str).str.strip().str.lower()
    done = ann[ann["my_label"].isin(mod.LABELS)]
    unsure = ann[ann["my_label"] == "unsure"]
    blank = len(ann) - len(done) - len(unsure)
    if blank:
        print(f"WARNING: {blank} rows still blank; scoring the {len(done)} completed.")
    y = np.array([mod.LABELS.index(gold[i]) for i in done.index])
    p = np.array([mod.LABELS.index(l) for l in done["my_label"]])
    m = E.evaluate(y, p, n_boot=4000, k=k)
    print(f"\n{ds}:  n scored = {len(done)}   unsure = {len(unsure)}")
    print(f"  QWK vs gold  : {m['qwk']:.3f} [{m['qwk_ci'][0]:.3f}, {m['qwk_ci'][1]:.3f}]")
    print(f"  exact match  : {m['acc']:.3f}")
    print(f"  MAE          : {m['mae']:.3f}")
    print(f"  adjacent (<=1): {float((np.abs(y-p)<=1).mean()):.3f}")
    if "ideation" in ann.columns:
        idl = ann.loc[done.index, "ideation"].astype(str).str.strip().str.lower()
        top = len(mod.LABELS) - 1
        for rule, desc in (("passive", "passive ideation promoted to top class"),
                           ("active", "active ideation demoted one band")):
            alt = p.copy()
            if rule == "passive":
                alt[(idl == "passive").values] = top
            else:
                mask = (idl == "active").values & (alt == top)
                alt[mask] = top - 1
            q = E._core(y, alt, k)["qwk"]
            print(f"  boundary sensitivity ({desc}): QWK {q:.3f}")
    print("\n  confusion (rows = gold, cols = yours):")
    print("             " + "".join(f"{l[:4]:>7}" for l in mod.LABELS))
    for i, l in enumerate(mod.LABELS):
        print(f"  {l:<14}" + "".join(f"{m['confusion'][i][j]:>7}" for j in range(k)))
    best = {"depseverity": 0.526, "depsign": 0.228}[ds]
    print(f"\nBest model cell on this corpus: QWK {best:.3f} (full test split). The "
          "sample here is stratified, so harder than natural prevalence --- treat as "
          "indicative, and report as single-annotator and non-clinician.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "make"
    ds = sys.argv[2] if len(sys.argv) > 2 else "depseverity"
    {"make": make, "score": score}[cmd](ds)
