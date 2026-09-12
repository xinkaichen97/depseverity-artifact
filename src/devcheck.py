"""Dev-subset diagnostics: does C3 extraction carry enough signal to aggregate on?"""
import collections, json, sys
from pathlib import Path
import conditions as C

RUNS = Path(__file__).resolve().parents[1] / "runs"
ORDER = ["minimum", "mild", "moderate", "severe"]


def load(tag):
    p = RUNS / tag
    return [json.loads(l) for l in p.open()] if p.exists() else []


def label_report(recs, name):
    ok = [r for r in recs if r.get("pred")]
    print(f"\n### {name}  n={len(recs)}  parse_failures={len(recs)-len(ok)}")
    ct = collections.Counter((r["gold"], r["pred"]) for r in ok)
    print("gold \\ pred   " + "  ".join(f"{l[:4]:>5}" for l in ORDER))
    for g in ORDER:
        print(f"  {g:<11} " + "  ".join(f"{ct[(g,p)]:>5}" for p in ORDER))
    exact = sum(ct[(g, g)] for g in ORDER)
    print(f"  exact match: {exact}/{len(ok)}")


def c3_report(recs):
    ok = [r for r in recs if r.get("items")]
    print(f"\n### C3  n={len(recs)}  parse_failures={len(recs)-len(ok)}")
    if not ok:
        return
    # status distribution per item
    print(f"\n{'item':<16} {'present':>8} {'absent':>8} {'unclear':>8}")
    for key, _ in C.ITEMS:
        c = collections.Counter(r["items"][key]["status"] for r in ok)
        print(f"{key:<16} {c['present']:>8} {c['absent']:>8} {c['unclear']:>8}")

    # THE question: how many unclears on a typical post?
    unc = [sum(1 for k, _ in C.ITEMS if r["items"][k]["status"] == "unclear") for r in ok]
    pres = [sum(1 for k, _ in C.ITEMS if r["items"][k]["status"] == "present") for r in ok]
    print(f"\nunclear per post: median {sorted(unc)[len(unc)//2]}  mean {sum(unc)/len(unc):.1f}  max 9")
    print(f"present per post: median {sorted(pres)[len(pres)//2]}  mean {sum(pres)/len(pres):.1f}")

    # Does the present-count separate the classes? This is the whole C3 premise.
    print(f"\n{'gold':<10} {'n':>3} {'mean present':>13} {'mean unclear':>13}  present counts")
    for g in ORDER:
        sub = [r for r in ok if r["gold"] == g]
        if not sub:
            continue
        p = [sum(1 for k, _ in C.ITEMS if r["items"][k]["status"] == "present") for r in sub]
        u = [sum(1 for k, _ in C.ITEMS if r["items"][k]["status"] == "unclear") for r in sub]
        print(f"{g:<10} {len(sub):>3} {sum(p)/len(p):>13.2f} {sum(u)/len(u):>13.2f}  {sorted(p)}")

    # item 9 vs gold severe
    print("\nself_harm fires by gold class:")
    for g in ORDER:
        sub = [r for r in ok if r["gold"] == g]
        if sub:
            n = sum(1 for r in sub if r["items"]["self_harm"]["status"] == "present")
            print(f"  {g:<10} {n}/{len(sub)}")

    # span grounding
    tot = sum(len(r.get("grounding") or {}) for r in ok)
    good = sum(sum(1 for v in (r.get("grounding") or {}).values() if v) for r in ok)
    print(f"\nspan grounding: {good}/{tot} verbatim spans found in post"
          + (f" ({good/tot*100:.0f}%)" if tot else ""))


m = sys.argv[1] if len(sys.argv) > 1 else "ollama-qwen3-8b"
for cond in ("C1", "C2"):
    r = load(f"{cond}_{m}_dev_seed0.jsonl")
    if r:
        label_report(r, cond)
r = load(f"C3_{m}_dev_seed0.jsonl")
if r:
    c3_report(r)
