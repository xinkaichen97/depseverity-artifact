"""The three experimental conditions.

C1 direct label, C2 chain-of-thought then label, C3 structured PHQ-9 extraction
with no label at all (aggregate.py assigns the label deterministically).
"""
from __future__ import annotations

import json
import re

LABELS = ["minimum", "mild", "moderate", "severe"]
LABEL_ALIASES = {"minimal": "minimum", "min": "minimum"}

ITEMS = [
    ("anhedonia", "little interest or pleasure in doing things"),
    ("depressed_mood", "feeling down, depressed, or hopeless"),
    ("sleep", "trouble falling or staying asleep, or sleeping too much"),
    ("fatigue", "feeling tired or having little energy"),
    ("appetite", "poor appetite, overeating, or weight change"),
    ("worthlessness", "feeling bad about yourself, worthless, or excessively guilty"),
    ("concentration", "trouble concentrating on things"),
    ("psychomotor", "moving or speaking noticeably slowly, or being restless and fidgety"),
    ("self_harm", "thoughts that you would be better off dead, or of hurting yourself"),
]

# --------------------------------------------------------------------------- C1

C1_SYSTEM = (
    "You are assessing the severity of depression expressed in a single social media post.\n"
    "Reply with exactly one word, the severity level, and nothing else.\n"
    "The permitted levels, from lowest to highest, are: minimum, mild, moderate, severe."
)
C1_USER = "Post:\n\"\"\"\n{text}\n\"\"\"\n\nSeverity level:"

# --------------------------------------------------------------------------- C2

C2_SYSTEM = (
    "You are assessing the severity of depression expressed in a single social media post.\n\n"
    "First reason step by step about which depressive symptoms are and are not evident in "
    "the post, and how strongly each is expressed. Then state the overall severity level.\n\n"
    "The permitted levels, from lowest to highest, are: minimum, mild, moderate, severe.\n\n"
    "End your reply with a final line in exactly this form:\n"
    "FINAL: <level>"
)
C2_USER = "Post:\n\"\"\"\n{text}\n\"\"\""

# --------------------------------------------------------------------------- C3
# The model never sees the severity levels and never produces one.

_ITEM_LINES = "\n".join(f'  {i+1}. "{k}" - {d}' for i, (k, d) in enumerate(ITEMS))

C3_SYSTEM = (
    "You are annotating a single social media post for the presence of nine specific "
    "symptoms. You are not rating, scoring, or diagnosing anything - you only report, for "
    "each symptom, whether the post gives evidence of it.\n\n"
    f"The nine symptoms are:\n{_ITEM_LINES}\n\n"
    "For each symptom return one of:\n"
    '  "present" - the post gives positive evidence that the writer experiences it\n'
    '  "absent"  - the post gives positive evidence that the writer does NOT experience it\n'
    '  "unclear" - the post does not say either way\n\n'
    'Use "unclear" when the post is simply silent about a symptom. Do not use "absent" '
    "merely because a symptom is unmentioned.\n\n"
    'When and only when a symptom is "present", also return "evidence": a short span '
    "copied verbatim from the post. Copy it exactly; do not paraphrase.\n\n"
    "Reply with JSON only - no preamble, no code fence, no commentary. Schema:\n"
    '{"<symptom>": {"status": "present|absent|unclear", "evidence": "<verbatim span or empty>"}, ...}\n'
    "Include all nine symptom keys exactly as written above."
)
C3_USER = "Post:\n\"\"\"\n{text}\n\"\"\""

# --------------------------------------------------------------------------- C4
# Same task and schema as C3, but the BDI-II 21-item inventory - the instrument the
# DepSeverity labels were actually assigned with. Pre-registered; see notes/findings.md.

BDI_ITEMS = [
    ("sadness", "feeling sad or unhappy"),
    ("pessimism", "feeling discouraged or hopeless about the future"),
    ("past_failure", "feeling like a failure, or dwelling on past failures"),
    ("loss_of_pleasure", "getting less pleasure from things previously enjoyed"),
    ("guilt", "feeling guilty"),
    ("punishment", "feeling one is being punished, or deserves punishment"),
    ("self_dislike", "disliking oneself, or having lost confidence in oneself"),
    ("self_criticism", "blaming or criticising oneself"),
    ("suicidal", "thoughts of killing oneself, or of being better off dead"),
    ("crying", "crying, or being unable to cry when one wants to"),
    ("agitation", "feeling restless, agitated, or keyed up"),
    ("loss_of_interest", "having lost interest in other people or activities"),
    ("indecisiveness", "finding it harder than usual to make decisions"),
    ("worthlessness", "feeling worthless, or of no value"),
    ("loss_of_energy", "having less energy than usual"),
    ("sleep_change", "sleeping more or less than usual, or broken sleep"),
    ("irritability", "being more irritable than usual"),
    ("appetite_change", "eating more or less than usual, or appetite change"),
    ("concentration", "finding it harder than usual to concentrate"),
    ("fatigue", "being too tired to do many of the things one used to do"),
    ("loss_of_interest_sex", "reduced interest in sex"),
]

_BDI_LINES = "\n".join(f'  {i+1}. "{k}" - {d}' for i, (k, d) in enumerate(BDI_ITEMS))

C4_SYSTEM = C3_SYSTEM.replace(
    "nine specific symptoms", "twenty-one specific symptoms"
).replace(
    f"The nine symptoms are:\n{_ITEM_LINES}",
    f"The twenty-one symptoms are:\n{_BDI_LINES}"
).replace(
    "Include all nine symptom keys exactly as written above.",
    "Include all twenty-one symptom keys exactly as written above.")
C4_USER = C3_USER


# --------------------------------------------------------------------------- C3 ablations
# Two prompt variants, to test whether the sparse extraction and the collapse of the
# three-valued scheme are properties of the corpus or of our prompt wording. Everything
# else is held fixed: same nine items, same JSON schema, same prohibition on the model
# seeing or emitting a severity label.

# C3P - permissive. Lowers the bar for "present" from stated evidence to reasonable
# implication, which is the form the "your prompt was too conservative" objection takes.
C3P_SYSTEM = (
    "You are annotating a single social media post for the presence of nine specific "
    "symptoms. You are not rating, scoring, or diagnosing anything - you only report, for "
    "each symptom, whether the post indicates it.\n\n"
    f"The nine symptoms are:\n{_ITEM_LINES}\n\n"
    "For each symptom return one of:\n"
    '  "present" - the post indicates the writer experiences it. Count it as present if '
    "the writer states it directly, describes it in their own words, or describes "
    "circumstances or behaviour from which it reasonably follows. Do not require a "
    "clinical phrasing, and do not require the writer to name the symptom.\n"
    '  "absent"  - the post indicates the writer does NOT experience it\n'
    '  "unclear" - the post gives no indication either way\n\n'
    "Err toward \"present\" when a reading of the post supports it. Reserve \"unclear\" "
    "for symptoms the post genuinely does not touch on.\n\n"
    'When a symptom is "present", also return "evidence": a short span copied verbatim '
    "from the post that supports the judgement. Copy it exactly; do not paraphrase.\n\n"
    "Reply with JSON only - no preamble, no code fence, no commentary. Schema:\n"
    '{"<symptom>": {"status": "present|absent|unclear", "evidence": "<verbatim span or empty>"}, ...}\n'
    "Include all nine symptom keys exactly as written above."
)

# C3S - symmetric evidence. Requires a verbatim span for "absent" as well as "present",
# so "absent" carries the same burden as "present". Tests whether the collapse of the
# three-valued scheme (absent used in 0.3% of judgements) is an artifact of the original
# prompt asking for evidence in only one direction.
C3S_SYSTEM = (
    "You are annotating a single social media post for the presence of nine specific "
    "symptoms. You are not rating, scoring, or diagnosing anything - you only report, for "
    "each symptom, what the post says about it.\n\n"
    f"The nine symptoms are:\n{_ITEM_LINES}\n\n"
    "For each symptom return one of:\n"
    '  "present" - the post gives evidence that the writer experiences it\n'
    '  "absent"  - the post gives evidence that the writer does NOT experience it. This '
    "includes the writer denying the symptom, describing its opposite, or describing "
    "functioning that is incompatible with it.\n"
    '  "unclear" - the post does not address the symptom\n\n'
    'For BOTH "present" and "absent" you must return "evidence": a short span copied '
    "verbatim from the post that supports the judgement. If you cannot quote a supporting "
    'span, the correct answer is "unclear". Copy spans exactly; do not paraphrase.\n\n'
    "Reply with JSON only - no preamble, no code fence, no commentary. Schema:\n"
    '{"<symptom>": {"status": "present|absent|unclear", "evidence": "<verbatim span or empty>"}, ...}\n'
    "Include all nine symptom keys exactly as written above."
)

SPECS = {
    "C1": (C1_SYSTEM, C1_USER),
    "C2": (C2_SYSTEM, C2_USER),
    "C3": (C3_SYSTEM, C3_USER),
    "C4": (C4_SYSTEM, C4_USER),
    "C3P": (C3P_SYSTEM, C3_USER),
    "C3S": (C3S_SYSTEM, C3_USER),
}

ITEMS_FOR = {"C3": ITEMS, "C4": BDI_ITEMS, "C3P": ITEMS, "C3S": ITEMS}


def prompt(condition: str, text: str, labels=None) -> tuple[str, str]:
    """C3/C4 never name the severity labels, so they are unaffected by `labels`."""
    sys_p, user_p = SPECS[condition]
    if labels is not None and condition in ("C1", "C2"):
        sys_p = sys_p.replace(
            "minimum, mild, moderate, severe", ", ".join(labels))
    return sys_p, user_p.format(text=text)


# ------------------------------------------------------------------- parsing

_LABEL_RE = re.compile(r"\b(minimum|minimal|mild|moderate|severe)\b", re.I)


def parse_label(text: str, last: bool, labels=None) -> str | None:
    """C1 takes the only label; C2 takes the final one."""
    labels = LABELS if labels is None else labels
    rx = re.compile("(" + "|".join(re.escape(l) for l in
                                   sorted(labels, key=len, reverse=True)) + ")", re.I)
    m = re.search(r"FINAL:\s*(.+)", text, re.I)
    if m:
        hit = rx.search(m.group(1))
        if hit:
            return hit.group(1).lower()
        cand = LABEL_ALIASES.get(m.group(1).strip().lower())
        if cand in labels:
            return cand
    hits = rx.findall(text)
    if not hits:
        alt = _LABEL_RE.findall(text)     # tolerate "minimal" etc.
        cand = LABEL_ALIASES.get(alt[-1].lower()) if alt else None
        return cand if cand in labels else None
    return hits[-1 if last else 0].lower()


_JSON_RE = re.compile(r"\{.*\}", re.S)
VALID_STATUS = {"present", "absent", "unclear"}


def parse_items(text: str, items=None) -> tuple[dict | None, str | None]:
    """Return (items, failure_reason). items maps key -> {status, evidence}."""
    items = ITEMS if items is None else items
    body = text.strip()
    body = re.sub(r"^```(?:json)?|```$", "", body, flags=re.M).strip()
    m = _JSON_RE.search(body)
    if not m:
        return None, "no_json_object"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"json_decode: {e.msg}"
    if not isinstance(obj, dict):
        return None, "not_an_object"

    out, missing = {}, []
    for key, _ in items:
        v = obj.get(key)
        if v is None:
            missing.append(key)
            continue
        if isinstance(v, str):           # tolerate bare "present"
            v = {"status": v, "evidence": ""}
        if not isinstance(v, dict) or "status" not in v:
            return None, f"bad_item_shape: {key}"
        status = str(v["status"]).strip().lower()
        if status not in VALID_STATUS:
            return None, f"bad_status: {key}={status!r}"
        out[key] = {"status": status, "evidence": str(v.get("evidence") or "")}
    if missing:
        return None, f"missing_items: {','.join(missing)}"
    return out, None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def span_grounded(evidence: str, post: str) -> bool | None:
    """Is the quoted evidence actually in the post? None when there is nothing to check."""
    if not evidence.strip():
        return None
    return _norm(evidence) in _norm(post)
