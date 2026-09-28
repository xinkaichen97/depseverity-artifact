"""Write the extended version's appendix tables (appendix_tables.tex) from stored results.

Inputs: numbers.json, per_class_recall.json (run per_class_recall.py first), the run records
in ../runs or ../runs_scrubbed (output lengths), ../results/*.md, and the prompt text in
../src/conditions.py. Nothing is transcribed by hand except the Holm survivor list, which
is the one reported in the paper's Limitations section.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import conditions as C  # noqa: E402

N = json.load(open(HERE / "numbers.json"))
R = json.load(open(HERE / "per_class_recall.json"))
MODELS = ["Qwen3.5-9B", "DeepSeek-V4.1-Flash", "Claude-Sonnet-5"]
SHORTM = {"Qwen3.5-9B": "Qwen3.5-9B", "DeepSeek-V4.1-Flash": "DeepSeek", "Claude-Sonnet-5": "Claude"}
RUNID = {"Qwen3.5-9B": "ollama-qwen3.5-9b", "DeepSeek-V4.1-Flash": "deepseek-deepseek-flash",
         "Claude-Sonnet-5": "anthropic-claude-sonnet-5"}
CORP = {"depseverity": "DepSeverity", "depsign": "DepSign"}
RUNS = HERE.parent / "runs" if (HERE.parent / "runs").is_dir() else HERE.parent / "runs_scrubbed"
# Holm survivors of the 18-comparison confirmatory family (paper, Limitations section).
HOLM = {("depseverity", "DeepSeek-V4.1-Flash", "C2 - C1"), ("depseverity", "Claude-Sonnet-5", "C2 - C1"),
        ("depsign", "DeepSeek-V4.1-Flash", "C2 - C1"), ("depsign", "Claude-Sonnet-5", "C2 - C1"),
        ("depseverity", "Qwen3.5-9B", "C3 fitted - C2"), ("depsign", "Qwen3.5-9B", "C2 - C1"),
        ("depsign", "Qwen3.5-9B", "C3 fitted - C2"), ("depsign", "Qwen3.5-9B", "C3 a priori - C2")}
CONFIRM = ("C2 - C1", "C3 fitted - C2", "C3 a priori - C2")
out = []


def f(x):
    return f"{x:+.3f}".replace("-", "$-$")


# Table A: every paired comparison with C2 or C1.
out += [r"\begin{table}[!htb]", r"\caption{All paired comparisons ($\Delta\kappa_w$, 95\% paired "
        r"bootstrap interval, 4000 resamples). Rows marked C are the confirmatory family; "
        r"H = survives Holm correction at $\alpha=0.05$. C4 uses the DSM-5 rule for \emph{a priori}.}",
        r"\label{tab:allcomp}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llllc}", r"\toprule",
        r"\textbf{Corpus} & \textbf{Model} & \textbf{Comparison} & $\Delta\kappa_w$ [95\% CI] & \\",
        r"\midrule"]
for ds in CORP:
    for m in MODELS:
        for comp in CONFIRM + ("C4 fitted - C2", "C4 a priori - C2", "C3 fitted - C1",
                               "C3 a priori - C1", "C4 fitted - C1", "C4 a priori - C1"):
            v = N[ds]["deltas"].get(f"{m}|{comp}")
            if v is None:
                continue
            mark = ("C" + (",H" if (ds, m, comp) in HOLM else "")) if comp in CONFIRM else ""
            d = f(v["d"]) if not v["sig"] else r"\textbf{" + f(v["d"]) + "}"
            out.append(f"{CORP[ds]} & {SHORTM[m]} & {comp.replace(' - ', ' $-$ ').replace('a priori', r'\emph{a priori}')} & "
                       f"{d} [{f(v['ci'][0])}, {f(v['ci'][1])}] & {mark} \\\\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# Table B: thresholds actually used.
out += [r"\begin{table}[!htb]", r"\caption{Thresholds on the count of \texttt{present} criteria "
        r"(C3: 0--9, C4: 0--21). Fitted thresholds maximize $\kappa_w$ on the 600-post fitting split.}",
        r"\label{tab:cutoffs}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lllll}", r"\toprule",
        r"\textbf{Corpus} & \textbf{Model} & \textbf{Cond.} & \textbf{fitted} & \textbf{\emph{a priori}} \\",
        r"\midrule"]
for ds in CORP:
    for m in MODELS:
        for c in ("C3", "C4"):
            a, b = N[ds]["cells"].get(f"{m}|{c} fitted"), N[ds]["cells"].get(f"{m}|{c} a priori")
            if a is None:
                continue
            fm = lambda cs: "[" + ", ".join(f"{x:g}" for x in cs) + "]"
            out.append(f"{CORP[ds]} & {SHORTM[m]} & {c} & {fm(a['cutoffs'])} & {fm(b['cutoffs'])} \\\\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# Table C: per-class recall.
for ds in CORP:
    labs = R[ds]["labels"]
    head = " & ".join(r"\textsc{" + {"not depression": "not dep."}.get(l, l) + "}" for l in labs)
    out += [r"\begin{table}[!htb]", rf"\caption{{Per-class recall on {CORP[ds]} test "
            rf"($n$ per class: {', '.join(str(x) for x in next(iter(R[ds]['cells'].values()))['n'])}).}}",
            rf"\label{{tab:recall-{ds}}}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{ll" + "c" * len(labs) + "}", r"\toprule",
            rf"\textbf{{Model}} & \textbf{{Cond.}} & {head} \\", r"\midrule"]
    for m in MODELS:
        first = True
        for key, v in R[ds]["cells"].items():
            if not key.startswith(m + "|"):
                continue
            tag = key.split("|")[1].replace("a priori", r"\emph{a pr.}")
            out.append(f"{SHORTM[m] if first else ''} & {tag} & "
                       + " & ".join(f"{x:.2f}" for x in v["recall"]) + r" \\")
            first = False
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# Table D: mean input and output tokens per call.
out += [r"\begin{table}[!htb]", r"\caption{Mean tokens per test call, input / output, as "
        r"reported by each provider.}", r"\label{tab:tokens}", r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llcccc}", r"\toprule",
        r"\textbf{Corpus} & \textbf{Model} & \textbf{C1} & \textbf{C2} & \textbf{C3} & \textbf{C4} \\",
        r"\midrule"]
for ds in CORP:
    for m in MODELS:
        cells = []
        for c in ("C1", "C2", "C3", "C4"):
            p = RUNS / f"{'' if ds == 'depseverity' else 'depsign_'}{c}_{RUNID[m]}_test_seed0.jsonl"
            if not p.exists():
                cells.append("--")
                continue
            recs = [json.loads(line)["meta"] for line in open(p)]
            cells.append(f"{st.mean(r['in_tokens'] for r in recs):.0f} / "
                         f"{st.mean(r['out_tokens'] for r in recs):.0f}")
        out.append(f"{CORP[ds]} & {SHORTM[m]} & " + " & ".join(cells) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]


# Prompts, verbatim, wrapped (DepSeverity label set; DepSign swaps in its three labels).
def tt(s):
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("{", r"\{"), ("}", r"\}"), ("_", r"\_"), ("#", r"\#"), ("%", r"\%"),
                 ("&", r"\&"), ("$", r"\$"), ("<", r"\textless{}"), (">", r"\textgreater{}"),
                 ("|", r"\textbar{}"), ('"', r"\textquotedbl{}"), ("~", r"\textasciitilde{}")):
        s = s.replace(a, b)
    return s


def prompt_block(title, text):
    lines = [tt(l) if l.strip() else "" for l in text.strip("\n").split("\n")]
    body = r"\\" "\n".join(l if l else r"\mbox{}" for l in lines)
    return [rf"\noindent\textbf{{{title}}}\par\nobreak\smallskip\nobreak",
            r"{\ttfamily\scriptsize\raggedright\noindent " + body + r"\par}\medskip", ""]



def md_rows(path, header_start):
    """Rows of the markdown table whose header line starts with `header_start`."""
    lines = open(path).read().split("\n")
    i = next(j for j, l in enumerate(lines) if l.startswith(header_start))
    rows = []
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip() for c in l.strip("|").split("|")])
    return rows


def md2tex(c):
    c = c.replace("**", "").replace("`", "").replace("$-$", "-")
    c = re.sub(r"(?<![\w.])-(?=\d)", "$-$", c)
    return c.replace("_", r"\_")


RES = HERE.parent / "results"
MN = {"qwen3.5": "Qwen3.5-9B", "deepseek-fla": "DeepSeek", "claude-sonne": "Claude"}
cal = md_rows(RES / "calibration_control.md", "| Corpus | Model | Cond. | raw")
cmp_ = {(r[0], r[1], r[2].split()[-1]): r[3] for r in
        md_rows(RES / "calibration_control.md", "| Corpus | Model | Comparison")}
out += [r"\begin{table}[!htb]", r"\caption{C1 and C2 given the same labels as C3: a monotone "
        r"relabeling fitted on the 600-post fitting split. The map lists, for each predicted label "
        r"from lowest to highest, the label it becomes (\texttt{0123} leaves labels unchanged). "
        r"The last column is C3 fitted minus the recalibrated condition.}",
        r"\label{tab:cal}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lllcccl}", r"\toprule",
        r"\textbf{Corpus} & \textbf{Model} & & raw & +cal & map & C3 fitted $-$ +cal \\", r"\midrule"]
for ds, m, cond, raw, calq, mp, _c3 in cal:
    d = cmp_[(ds, m, cond + "+cal")]
    out.append(f"{CORP[ds]} & {MN[m]} & {cond} & {raw} & {md2tex(calq)} & \\texttt{{{mp.strip('`')}}} & "
               f"{md2tex(d.replace('$', ''))} \\\\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

abl = md_rows(RES / "prompt_ablation.md", "| Variant | present/post")
ablc = {r[0]: r[1:] for r in md_rows(RES / "prompt_ablation.md", "| Variant | vs C2")}
out += [r"\begin{table}[!htb]", r"\caption{C3 prompt variants, DeepSeek-V4.1-Flash on DepSeverity "
        r"(same nine criteria and JSON schema; wording only). Last two columns: $\Delta\kappa_w$ "
        r"against C2.}", r"\label{tab:ablation}", r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{lcccccc}", r"\toprule",
        r"\textbf{Variant} & pres./post & \texttt{absent} & fitted & \emph{a pr.} & "
        r"vs C2, fitted & vs C2, \emph{a pr.} \\", r"\midrule"]
for v, pp, ab, _g, qf, qa in abl:
    cf, ca = (md2tex(x.replace("$", "")).replace(" ns", "").replace(" sig", "") for x in ablc[v])
    out.append(f"{v} & {pp} & {ab.replace('%', chr(92) + '%')} & {qf} & {qa} & {cf} & {ca} \\\\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

(HERE / "appendix_tables.tex").write_text("\n".join(out) + "\n")
NAMES = {"tab:allcomp": "comp", "tab:cutoffs": "cutoffs", "tab:cal": "cal", "tab:tokens": "tokens",
         "tab:ablation": "ablation", "tab:recall-depseverity": "recall", "tab:recall-depsign": "recall"}
parts = {}
for block in "\n".join(out).split(r"\begin{table}")[1:]:
    label = re.search(r"\\label\{([^}]+)\}", block)[1]
    text = r"\begin{table}" + block.rstrip() + "\n"
    if NAMES[label] in ("comp", "cal", "ablation"):  # too wide for one column
        text = text.replace(r"\begin{table}[!htb]", r"\begin{table*}[!t]").replace(
            r"\end{table}", r"\end{table*}")
    parts.setdefault(NAMES[label], []).append(text)
for name, blocks in parts.items():
    (HERE / f"appendix_tables_{name}.tex").write_text("\n".join(blocks))

P = []
for cond, title in (("C1", "C1 (direct)"), ("C2", "C2 (chain-of-thought)"),
                    ("C3", "C3 (PHQ-9 extraction)"), ("C4", "C4 (BDI-II extraction)"),
                    ("C3P", "C3P (permissive variant)"), ("C3S", "C3S (symmetric variant)")):
    system, user = C.SPECS[cond]
    P += prompt_block(f"{title}, system prompt", system)
    if cond in ("C1", "C2", "C3"):
        P += prompt_block(f"{title}, user prompt", user)
(HERE / "appendix_prompts.tex").write_text("\n".join(P) + "\n")
print(f"wrote appendix_tables.tex ({len(out)} lines), appendix_prompts.tex ({len(P)} lines)")
