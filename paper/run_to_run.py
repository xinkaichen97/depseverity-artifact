"""Run-to-run variance for Claude-Sonnet-5 and DeepSeek-V4.1-Flash (Sec. V-F).
Run from depseverity/ with PYTHONPATH=src and the project venv."""
import json, math
import numpy as np
import aggregate as A, evaluate as E, data as D

N = json.load(open("paper/numbers.json"))
LAB = D.LABELS
MODELS = {"Claude-Sonnet-5": "anthropic:claude-sonnet-5", "DeepSeek-V4.1-Flash": "deepseek:deepseek-flash"}

def c2_labels(model, seed):
    return {r["id"]: (LAB.index(r["gold"]), LAB.index(r["pred"]))
            for r in A.load_run("C2", model, "test", seed=seed) if r.get("pred")}

def c3_labels(model, seed, cuts):
    recs = [r for r in A.load_run("C3", model, "test", seed=seed) if r.get("items")]
    s = np.array([A.score(r["items"]) for r in recs])
    p = A.apply_cutoffs(s, cuts)
    return {r["id"]: (LAB.index(r["gold"]), int(q)) for r, q in zip(recs, p)}

def kw(d):
    return E._core(np.array([v[0] for v in d.values()]), np.array([v[1] for v in d.values()]), 4)["qwk"]

def compare(a, b):
    ids = sorted(set(a) & set(b))
    return abs(kw({i: a[i] for i in ids}) - kw({i: b[i] for i in ids})), sum(a[i][1] != b[i][1] for i in ids), len(ids)

sd = {}
for short, model in MODELS.items():
    n1 = len(A.load_run("C2", model, "test", seed=1)), len(A.load_run("C3", model, "test", seed=1))
    if min(n1) < 706:
        print(f"{short}: seed-1 runs incomplete {n1}; skipping"); continue
    cuts0 = N["depseverity"]["cells"][f"{short}|C3 fitted"]["cutoffs"]
    dC2, chC2, n = compare(c2_labels(model, 0), c2_labels(model, 1))
    dC3, chC3, _ = compare(c3_labels(model, 0, cuts0), c3_labels(model, 1, cuts0))
    line = f"{short:20} C2: |dkw|={dC2:.3f}, labels changed {chC2}/{n} | C3 (seed-0 cutoffs {cuts0}): |dkw|={dC3:.3f}, changed {chC3}/{n}"
    if short == "Claude-Sonnet-5":
        try:
            b1 = A.build(model, seed=1)
            c3b = {i: (t, p) for i, t, p in zip(b1["ids"], b1["y_true"], b1["y_pred"])}
            dC3b, chC3b, _ = compare(c3_labels(model, 0, cuts0), c3b)
            line += f" | C3 with seed-1 refit cutoffs {b1['cutoffs']}: |dkw|={dC3b:.3f}, changed {chC3b}"
        except SystemExit as e:
            line += f" | (no seed-1 fit run: {e})"
    print(line)
    m = max(dC2, dC3); sd[short] = {"standard": m / math.sqrt(2), "conservative": m}

if "DeepSeek-V4.1-Flash" not in sd:
    raise SystemExit("DeepSeek repeat not available yet")
print("\nper-run SD used:", {k: {s: round(v, 4) for s, v in d.items()} for k, d in sd.items()})

rows = []
for ds in ("depseverity", "depsign"):
    for key, v in N[ds]["deltas"].items():
        model = key.split("|")[0]
        if model in sd:
            rows.append((f"{ds[:6]} {key}", v["d"], *v["ci"], v["sig"], model, 2))
rows += [("depsev DeepSeek C3 fitted - metadata", 0.122, 0.040, 0.201, True, "DeepSeek-V4.1-Flash", 1),
         ("depsev Claude C3 fitted - metadata", 0.095, 0.014, 0.176, True, "Claude-Sonnet-5", 1)]
print(f"\n{'comparison':46} {'published':22} {'standard':22} {'conservative':22}")
for name, dv, lo, hi, sig, model, nterms in rows:
    out = []
    for setting in ("standard", "conservative"):
        se = (hi - lo) / (2 * 1.96); g = math.sqrt(nterms) * sd[model][setting]
        s2 = math.sqrt(se ** 2 + g ** 2); l2, h2 = dv - 1.96 * s2, dv + 1.96 * s2
        out.append((l2, h2, l2 > 0 or h2 < 0))
    if sig or any(o[2] for o in out):
        flag = "" if all(o[2] == sig for o in out) else "  <-- changes"
        print(f"{name:46} [{lo:+.3f},{hi:+.3f}] {'SIG' if sig else 'ns '}  " +
              "  ".join(f"[{o[0]:+.3f},{o[1]:+.3f}] {'SIG' if o[2] else 'ns '}" for o in out) + flag)
