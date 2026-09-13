"""Execute one (condition, model, seed) over a subset. One JSONL per run."""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import conditions as C
import data as D
import data_depsign as DS
import providers as P

DATASETS = {"depseverity": (D, D.LABELS), "depsign": (DS, DS.LABELS)}

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def subset(df, which: str):
    if which == "dev":
        return df[df.is_dev]
    if which == "fit":
        return df[df.is_fit]
    if which == "train":
        return df[(df.split == "train") & (~df.is_dev)]
    if which == "test":
        return df[df.split == "test"]
    raise ValueError(which)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["C1", "C2", "C3", "C4", "C3P", "C3S"])
    ap.add_argument("--model", required=True, help="anthropic:claude-sonnet-5 | ollama:qwen3:8b")
    ap.add_argument("--subset", default="dev", choices=["dev", "fit", "train", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dataset", default="depseverity", choices=list(DATASETS))
    ap.add_argument("--workers", type=int, default=None,
                    help="concurrent requests; defaults to 8 for hosted APIs and 1 "
                         "for ollama (one local model, serial by necessity)")
    args = ap.parse_args()

    mod, labels = DATASETS[args.dataset]
    prov = P.build(args.model)
    workers = args.workers if args.workers else (1 if args.model.startswith("ollama") else 8)
    df = subset(mod.load(), args.subset)
    if args.limit:
        df = df.head(args.limit)

    safe = re.sub(r"[^A-Za-z0-9._-]", "-", args.model)
    prefix = "" if args.dataset == "depseverity" else f"{args.dataset}_"
    tag = f"{prefix}{args.condition}_{safe}_{args.subset}_seed{args.seed}"
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"{tag}.jsonl"

    def work(item):
        idx, row = item
        sys_p, user_p = C.prompt(args.condition, row.text, labels=labels)
        rec = {"id": int(idx), "gold": row.label, "condition": args.condition,
               "model": args.model, "seed": args.seed}
        try:
            r = prov.complete(sys_p, user_p, args.seed)
        except P.ProviderError as e:
            return rec | {"error": str(e)}

        rec["raw"] = r.text          # verbatim, always
        rec["meta"] = r.meta
        rec["cached"] = r.cached

        if args.condition in ("C3", "C4", "C3P", "C3S"):
            items, why = C.parse_items(r.text, C.ITEMS_FOR[args.condition])
            rec["items"] = items
            rec["parse_error"] = why
            if items is not None:
                rec["grounding"] = {
                    k: C.span_grounded(v["evidence"], row.text)
                    for k, v in items.items() if v["evidence"].strip()
                }
        else:
            lab = C.parse_label(r.text, last=(args.condition == "C2"), labels=labels)
            rec["pred"] = lab
            rec["parse_error"] = None if lab else "no_label_found"
        return rec

    items = list(df.iterrows())
    if workers == 1:
        recs = []
        for i, it in enumerate(items, 1):
            recs.append(work(it))
            if i % 25 == 0 or i == len(items):
                print(f"[{i}/{len(items)}]")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            recs = list(ex.map(work, items))   # ordered: output is deterministic

    n_err = sum("error" in r for r in recs)
    n_fail = sum(r.get("parse_error") is not None for r in recs if "error" not in r)
    n_cached = sum(r.pop("cached", False) for r in recs)
    with out.open("w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{out}\n  n={len(df)}  workers={workers}  cached={n_cached}  "
          f"parse_failures={n_fail} ({n_fail/len(df)*100:.1f}%)  call_errors={n_err}")


if __name__ == "__main__":
    main()
