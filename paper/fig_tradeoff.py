"""Extended-version figure: ordinal agreement against missed SEVERE posts.

One point per model x condition on each corpus, read from numbers.json. Colour says who
makes the prediction: the model itself (C1, C2, C2S) or code counting marked criteria
(C3 and C4, fitted or a priori, and the C2S status lines counted a priori). An arrow runs from
each model's C2S prediction to the a priori count of the same status lines. Run from the
paper directory.
"""
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch, HandlerTuple
from matplotlib.patches import FancyArrow

N = json.load(open("numbers.json"))
MODELS = [("Qwen3.5-9B", "o"), ("DeepSeek-V4.1-Flash", "s"), ("Claude-Sonnet-5", "^")]
MODEL_LABEL = ["C1", "C2", "C2S"]
# C2S status lines appear only as their a priori count, the arrow target.
COUNTED = ["C3 fitted", "C3 a priori", "C4 fitted", "C4 a priori", "C2S counted a priori"]
# Charcoal against rose: a neutral and one accent, distinct in greyscale and for colour-blind
# readers, and neither is a model colour in Fig. 1.
PRED, COUNT, GREY = "#37474f", "#c2185b", "#9a9a9a"
CORPORA = [("depseverity", "DepSeverity"), ("depsign", "DepSign")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7, "axes.linewidth": 0.6,
    "text.usetex": False, "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # IEEE PDF eXpress rejects Type 3 fonts
})

# One column: DepSeverity above DepSign.
fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.4), sharey=True)
for ax, (ds, title) in zip(axes, CORPORA):
    cells = N[ds]["cells"]

    def point(model, cond):
        c = cells.get(f"{model}|{cond}")
        # Share of SEVERE posts missed: the same quantity as Table II's SEV column.
        return None if c is None else (c["qwk"], c["sev_missed"] / c["n_sev"])

    for model, mk in MODELS:
        a, b = point(model, "C2S"), point(model, "C2S counted a priori")
        if a and b:
            # Same C2S responses at both ends: the model's own prediction (tail) and
            # their status lines counted with the a priori rule (head).
            ax.annotate("", xy=b, xytext=a, zorder=1,
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.7,
                                        mutation_scale=7, shrinkA=3.5, shrinkB=3.5))
        for conds, col in ((MODEL_LABEL, PRED), (COUNTED, COUNT)):
            for cond in conds:
                p = point(model, cond)
                if p:
                    # Arrow endpoints on top, so no other point hides an arrow's target.
                    top = cond in ("C2S", "C2S counted a priori")
                    ax.scatter(*p, marker=mk, s=22, color=col, edgecolor="white",
                               linewidth=0.6, zorder=4 if top else 3)
    ax.set_title(title, fontsize=7.5, style="italic")
    ax.set_xlabel(r"$\kappa_w$")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
for ax in axes:
    ax.set_ylabel("share of SEVERE posts missed")


# A colour entry shows all three model markers in that colour ("any blue marker"), so the
# legend uses only glyphs that appear in the plot and no shape means two things.
def colour(col):
    return tuple(Line2D([], [], marker=mk, ls="", color=col, markeredgecolor="white",
                        markeredgewidth=0.6, markersize=4.5) for _, mk in MODELS)


def arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return FancyArrow(0, height / 2, width, 0, length_includes_head=True, width=0.6,
                      head_width=0.85 * height, head_length=0.6 * height, color=GREY, lw=0)


ARROW = FancyArrow(0, 0, 1, 0, color=GREY, lw=0)   # the legend copies colour from this proxy
enc = [colour(PRED), colour(COUNT), ARROW]
enc_labels = ["prediction made by the model: C1, C2, C2S",
              "prediction computed by counting criteria: C3, C4, C2S counted a priori",
              "same C2S response: model's prediction \u2192 its criteria counted a priori"]
# Hollow, so the model key reads as shape only and not as the charcoal colour class.
shapes = [Line2D([], [], marker=mk, ls="", markerfacecolor="white", markeredgecolor="#555555",
                 markeredgewidth=0.9, label=m) for m, mk in MODELS]
fig.legend(handles=enc, labels=enc_labels, loc="lower left", ncol=1, frameon=False,
           bbox_to_anchor=(0.02, 0.045),
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15),
                        FancyArrow: HandlerPatch(patch_func=arrow)},
           handlelength=2.6, handletextpad=0.5, labelspacing=0.35)
fig.legend(handles=shapes, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.17, 1, 1), h_pad=0.8)
fig.savefig("fig_tradeoff.pdf", bbox_inches="tight", pad_inches=0.01)
