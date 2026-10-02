"""Extended-version figure: ordinal agreement against missed SEVERE posts.

One point per model x condition on each corpus, read from numbers.json. Colour says who
assigns the label: the model itself (C1, C2, C2S) or code counting marked criteria
(C3, C4 and the C2S checklist, fitted or a priori). A grey line joins each C2S label to
the a priori count of the same status lines. Run from the paper directory.
"""
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

N = json.load(open("numbers.json"))
MODELS = [("Qwen3.5-9B", "o"), ("DeepSeek-V4.1-Flash", "s"), ("Claude-Sonnet-5", "^")]
MODEL_LABEL = ["C1", "C2", "C2S"]
COUNTED = ["C3 fitted", "C3 a priori", "C4 fitted", "C4 a priori",
           "C2S counted fitted", "C2S counted a priori"]
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9a9a"   # validated reference palette, slots 1-2
CORPORA = [("depseverity", "DepSeverity"), ("depsign", "DepSign")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7, "axes.linewidth": 0.6,
    "text.usetex": False, "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # IEEE PDF eXpress rejects Type 3 fonts
})

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
for ax, (ds, title) in zip(axes, CORPORA):
    cells = N[ds]["cells"]

    def point(model, cond):
        c = cells.get(f"{model}|{cond}")
        return None if c is None else (c["qwk"], c["sev_missed"] / c["n_sev"])

    for model, mk in MODELS:
        a, b = point(model, "C2S"), point(model, "C2S counted a priori")
        if a and b:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=GREY, lw=0.7, zorder=1)
        for conds, col in ((MODEL_LABEL, BLUE), (COUNTED, ORANGE)):
            for cond in conds:
                p = point(model, cond)
                if p:
                    ax.scatter(*p, marker=mk, s=22, color=col, edgecolor="white",
                               linewidth=0.6, zorder=3)
    ax.set_title(title, fontsize=7.5, style="italic")
    ax.set_xlabel(r"$\kappa_w$")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("share of SEVERE posts missed")

enc = [Line2D([], [], marker="o", ls="", color=BLUE, label="label from the model (C1, C2, C2S)"),
       Line2D([], [], marker="o", ls="", color=ORANGE,
              label="label from counting criteria (C3, C4, C2S counted)"),
       Line2D([], [], color=GREY, lw=0.7, label="C2S label vs. its own status lines, counted a priori")]
shapes = [Line2D([], [], marker=mk, ls="", color="#444444", label=m) for m, mk in MODELS]
fig.legend(handles=enc, loc="lower center", ncol=3, frameon=False,
                 bbox_to_anchor=(0.5, 0.055), handletextpad=0.3, columnspacing=1.6)
fig.legend(handles=shapes, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.6)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig("fig_tradeoff.pdf", bbox_inches="tight", pad_inches=0.01)
