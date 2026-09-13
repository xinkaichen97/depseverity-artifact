"""Worked examples for reviewers, carrying no post text.

The paper's claims are about how extracted criteria turn into a label, so each case
shows the per-item statuses, the score, and what each threshold regime does with it.
Post text is not redistributed (neither corpus is licensed); a reviewer with the corpora
can look each case up by its split id. Evidence spans are quoted from the post, so we
report only whether one was given.

No model calls: everything is read from the cached runs.
"""
import json

import aggregate as A
import conditions as C
import data as D

MODEL, DS = "deepseek:deepseek-flash", "depseverity"
KEYS = [k for k, _ in C.ITEMS]


def main():
    df = D.load()
    sub = df["subreddit"].to_dict()
    fit = A.build(MODEL, condition="C3", dataset=DS)
    apri = A.build(MODEL, condition="C3", cutoffs=A.CLINICAL_CUTOFFS, dataset=DS)
    c2 = {r["id"]: r["pred"] for r in A.load_run("C2", MODEL, "test", dataset=DS)
          if r.get("pred")}
    items = {r["id"]: r for r in A.load_run("C3", MODEL, "test", dataset=DS)
             if r.get("items")}

    L = D.LABELS
    rows = []
    for i, pid in enumerate(fit["ids"]):
        rows.append(dict(
            id=pid, subreddit=sub.get(pid, "?"), gold=L[fit["y_true"][i]],
            score=fit["scores"][i], c3_fitted=L[fit["y_pred"][i]],
            c3_apriori=L[apri["y_pred"][apri["ids"].index(pid)]],
            c2=c2.get(pid, "?"), rec=items[pid]))

    def pick(name, why, test, n=2):
        hits = [r for r in rows if test(r)][:n]
        return [(name, why, h) for h in hits]

    cases = (
        pick("severe missed at the aggregation floor",
             "gold severe, but too few items fire for any cutoff to reach severe",
             lambda r: r["gold"] == "severe" and r["score"] <= 1)
        + pick("threshold regime flips the label",
               "same extraction; fitted and a priori cutoffs disagree",
               lambda r: r["c3_fitted"] != r["c3_apriori"], 3)
        + pick("chain-of-thought right, structure wrong",
               "C2 matches gold, C3 does not, on identical input",
               lambda r: r["c2"] == r["gold"] and r["c3_fitted"] != r["gold"])
        + pick("item 9 as a standalone flag",
               "self-harm ideation fires on a gold-severe post",
               lambda r: r["rec"]["items"]["self_harm"]["status"] == "present"
               and r["gold"] == "severe")
        + pick("over-extraction on a minimum post",
               "many items fire where gold is the lowest class",
               lambda r: r["gold"] == "minimum" and r["score"] >= 4)
    )

    out = [f"# Worked examples --- {MODEL}, DepSeverity test split\n",
           "No post text: neither corpus is licensed, so cases are identified by split",
           "`id` for readers who hold the data. Fitted cutoffs "
           f"{[float(c) for c in fit['cutoffs']]}; "
           f"*a priori* {list(A.CLINICAL_CUTOFFS)}.\n"]
    for name, why, r in cases:
        rec = r["rec"]
        on = [f"{k}{'*' if rec['items'][k].get('evidence') else ''}"
              for k in KEYS if rec["items"][k]["status"] == "present"]
        unc = [k for k in KEYS if rec["items"][k]["status"] == "unclear"]
        out += [f"### {name} --- post `{r['id']}` (r/{r['subreddit']})", f"_{why}_\n",
                f"- gold **{r['gold']}** | C2 {r['c2']} | "
                f"C3 fitted **{r['c3_fitted']}** | C3 *a priori* **{r['c3_apriori']}**",
                f"- score {r['score']:g} of 9; present: {', '.join(on) or 'none'}",
                f"- unclear: {', '.join(unc) or 'none'} (`*` = evidence span returned)\n"]
    open("results/qualitative.md", "w").write("\n".join(out))
    print(f"{len(cases)} cases -> results/qualitative.md")


if __name__ == "__main__":
    main()
