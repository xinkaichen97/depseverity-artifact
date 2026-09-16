"""Figure 1: paired bootstrap Delta kappa_w, all cross-condition comparisons.
Reads numbers.json; replaces the former Table III."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = json.load(open("numbers.json"))

ROWS = [  # (corpus, model label, key prefix)
    ("DepSeverity", "Qwen3.5-9B",          "depseverity", "Qwen3.5-9B"),
    ("DepSeverity", "DeepSeek-V4.1-Flash", "depseverity", "DeepSeek-V4.1-Flash"),
    ("DepSeverity", "Claude-Sonnet-5",     "depseverity", "Claude-Sonnet-5"),
    ("DepSign",     "Qwen3.5-9B",          "depsign",     "Qwen3.5-9B"),
    ("DepSign",     "DeepSeek-V4.1-Flash", "depsign",     "DeepSeek-V4.1-Flash"),
    ("DepSign",     "Claude-Sonnet-5",     "depsign",     "Claude-Sonnet-5"),
]
SERIES = [
    ("C2 $-$ C1",                  "C2 - C1",            "o", "#1a1a1a"),
    ("C3 $-$ C2, fitted",          "C3 fitted - C2",     "s", "#1f5fa8"),
    ("C3 $-$ C2, $\\it{a\\ priori}$", "C3 a priori - C2", "D", "#c0392b"),
]
OFF = [0.26, 0.0, -0.26]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7, "axes.linewidth": 0.6,
    "text.usetex": False, "mathtext.fontset": "stix",
})

fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.axvline(0, color="#999999", lw=0.7, ls=(0, (3, 2)), zorder=1)

for r, (_, _, corpus, model) in enumerate(ROWS):
    y0 = len(ROWS) - 1 - r
    for (lab, key, mk, col), dy in zip(SERIES, OFF):
        e = D[corpus]["deltas"].get(f"{model}|{key}")
        if e is None:
            continue
        y = y0 + dy
        lo, hi = e["ci"]
        sig = e["sig"]
        # Significant effects are drawn solid at full opacity; non-significant ones are
        # hollow and faded, so the eye lands on the comparisons that exclude zero.
        a = 1.0 if sig else 0.35
        ax.plot([lo, hi], [y, y], color=col, lw=0.9, solid_capstyle="butt", zorder=2,
                alpha=a)
        ax.plot([lo, lo, hi, hi], [y - .09, y + .09, y - .09, y + .09], ls="none",
                marker="|", ms=0, color=col, alpha=a)
        ax.plot(e["d"], y, marker=mk, ms=3.4, color=col, zorder=3,
                mfc=col if sig else "white", mew=0.8, alpha=a)

ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([m for _, m, _, _ in ROWS][::-1])
ax.set_ylim(-0.55, len(ROWS) - 0.45)
ax.set_xlabel(r"$\Delta\kappa_w$   (paired bootstrap, 95% CI)", labelpad=2)
ax.tick_params(axis="both", length=2, pad=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# corpus bands
ax.axhline(2.5, color="#cccccc", lw=0.5)
ax.text(0.985, 0.985, "DepSeverity", transform=ax.transAxes, ha="right", va="top",
        fontsize=6.5, style="italic", color="#555555")
ax.text(0.985, 2.4, "DepSign", transform=ax.get_yaxis_transform(), ha="right", va="top",
        fontsize=6.5, style="italic", color="#555555")

# Proxy handles: the legend names the three series only, so every entry is drawn the
# same way. Reusing plotted artists would inherit one data point's significance fill.
handles = [Line2D([], [], color=col, marker=mk, ms=3.4, lw=0.9, mfc=col, mew=0.8,
                  label=lab) for lab, _, mk, col in SERIES]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
          fontsize=6.4, frameon=False, handlelength=1.1, borderpad=0.0,
          columnspacing=1.1, handletextpad=0.35)
fig.tight_layout(pad=0.25)
fig.savefig("fig_deltas.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig("/tmp/fig_preview.png", dpi=260, bbox_inches="tight", pad_inches=0.01)
print("wrote fig_deltas.pdf")
