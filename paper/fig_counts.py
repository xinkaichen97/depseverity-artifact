"""Extended-version figure: how many PHQ-9 criteria C3 marks present per test post.

Share of test posts at each count, per model and corpus, read from the C3 run records
(../runs, or ../runs_scrubbed in the artifact), with the a priori thresholds drawn in.
Malformed responses are repaired as in aggregate.py (statuses recovered from the raw text),
so the counts match the scored records in both ../runs and ../runs_scrubbed. Run from the
paper directory.
"""
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

RUNS = "../runs" if os.path.isdir("../runs") else "../runs_scrubbed"
MODELS = [("Qwen3.5-9B", "ollama-qwen3.5-9b", "#1baf7a"),
          ("DeepSeek-V4.1-Flash", "deepseek-deepseek-flash", "#2a78d6"),
          ("Claude-Sonnet-5", "anthropic-claude-sonnet-5", "#eb6834")]   # validated slots 1-3
# Both panels share one count axis, 0 to 7+ (the last bin pools every higher count), so the
# stacked panels line up count for count.
CORPORA = [("depseverity", "DepSeverity", [0.5, 2.5, 4.5], ["MINIMUM", "MILD", "MODERATE", "SEVERE"], 7),
           ("depsign", "DepSign", [0.5, 4.5], ["NOT DEP.", "MODERATE", "SEVERE"], 7)]


def statuses(r):
    """The record's statuses, recovering them from a malformed response as aggregate.py does."""
    if r.get("items"):
        return r["items"]
    out = {}
    for m in re.finditer(r'"(\w+)"\s*:\s*\{', r.get("raw") or ""):
        st = re.search(r'"status"\s*:\s*"(present|absent|unclear)"', r["raw"][m.end():])
        if st:
            out[m.group(1)] = {"status": st.group(1)}
    return out if len(out) == 9 else None

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7, "axes.linewidth": 0.6,
    "text.usetex": False, "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # IEEE PDF eXpress rejects Type 3 fonts
})

# One column: DepSeverity above DepSign.
fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.5), sharex=True, sharey=True)
w = 0.27
for ax, (ds, title, cuts, bands, MAXC) in zip(axes, CORPORA):
    pre = "" if ds == "depseverity" else "depsign_"
    for j, (name, tag, col) in enumerate(MODELS):
        recs = [json.loads(line) for line in open(f"{RUNS}/{pre}C3_{tag}_test_seed0.jsonl")]
        items = [statuses(r) for r in recs]
        n = [sum(v["status"] == "present" for v in i.values()) for i in items if i]
        share = np.bincount(np.minimum(n, MAXC), minlength=MAXC + 1) / len(n)
        ax.bar(np.arange(MAXC + 1) + (j - 1) * w, share, width=w - 0.03, color=col,
               label=name, zorder=2)
    edges = [-0.5] + cuts + [MAXC + 0.5]
    for c in cuts:
        ax.axvline(c, color="#444444", lw=0.7, ls=(0, (3, 2)), zorder=3)
    for lo, hi, b in zip(edges[:-1], edges[1:], bands):
        ax.text((lo + hi) / 2, 1.0, b, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=6, color="#52514e")
    ax.set_xticks(range(MAXC + 1), [str(i) for i in range(MAXC)] + [f"{MAXC}+"])
    ax.set_title(title, fontsize=7.5, style="italic", pad=10)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
for ax in axes:
    ax.set_ylabel("share of test posts")
axes[-1].set_xlabel("criteria marked present (C3)")
# DepSeverity's SEVERE band (counts 5-7+) is empty; centre the legend there so no threshold
# line runs through it.
lo, hi = axes[0].get_xlim()
axes[0].legend(frameon=False, loc="upper center", handlelength=1.0, fontsize=5.8,
               bbox_to_anchor=(((4.5 + 7.5) / 2 - lo) / (hi - lo), 0.9))
fig.tight_layout(h_pad=0.6)
fig.savefig("fig_counts.pdf", bbox_inches="tight", pad_inches=0.01)
