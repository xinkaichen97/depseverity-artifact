# Exact prompt text

Reproduced verbatim from `src/conditions.py`. `{text}` is the post.
C3P and C3S are the prompt ablation; they differ from C3 in wording
only, over the same nine items and the same JSON schema.


## C1

direct label, no reasoning step.

### system

```
You are assessing the severity of depression expressed in a single social media post.
Reply with exactly one word, the severity level, and nothing else.
The permitted levels, from lowest to highest, are: minimum, mild, moderate, severe.
```

### user

```
Post:
"""
{text}
"""

Severity level:
```

## C2

chain-of-thought, then the label; the model still emits the label.

### system

```
You are assessing the severity of depression expressed in a single social media post.

First reason step by step about which depressive symptoms are and are not evident in the post, and how strongly each is expressed. Then state the overall severity level.

The permitted levels, from lowest to highest, are: minimum, mild, moderate, severe.

End your reply with a final line in exactly this form:
FINAL: <level>
```

### user

```
Post:
"""
{text}
"""
```

## C2S

**ablation** — chain-of-thought that must first walk through the nine PHQ-9 criteria, one line each with a status; the model still emits the label. The status lines can also be counted with C3's rule ("C2S counted").

### system

```
You are assessing the severity of depression expressed in a single social media post.

First go through the nine symptoms below in order, one line each, in exactly this form:
<number>. <symptom>: present|absent|unclear - <brief reason>
Write each symptom name exactly as given below. Use "present" when the post gives evidence that the writer experiences the symptom, "absent" when it gives evidence that the writer does not, and "unclear" when it says neither.

The nine symptoms are:
  1. "anhedonia" - little interest or pleasure in doing things
  2. "depressed_mood" - feeling down, depressed, or hopeless
  3. "sleep" - trouble falling or staying asleep, or sleeping too much
  4. "fatigue" - feeling tired or having little energy
  5. "appetite" - poor appetite, overeating, or weight change
  6. "worthlessness" - feeling bad about yourself, worthless, or excessively guilty
  7. "concentration" - trouble concentrating on things
  8. "psychomotor" - moving or speaking noticeably slowly, or being restless and fidgety
  9. "self_harm" - thoughts that you would be better off dead, or of hurting yourself

Then reason about how severe the depression expressed in the post is overall, and state the overall severity level.

The permitted levels, from lowest to highest, are: minimum, mild, moderate, severe.

End your reply with a final line in exactly this form:
FINAL: <level>
```

### user

```
Post:
"""
{text}
"""
```

## C3

structured PHQ-9 extraction; the model never names a severity label.

### system

```
You are annotating a single social media post for the presence of nine specific symptoms. You are not rating, scoring, or diagnosing anything - you only report, for each symptom, whether the post gives evidence of it.

The nine symptoms are:
  1. "anhedonia" - little interest or pleasure in doing things
  2. "depressed_mood" - feeling down, depressed, or hopeless
  3. "sleep" - trouble falling or staying asleep, or sleeping too much
  4. "fatigue" - feeling tired or having little energy
  5. "appetite" - poor appetite, overeating, or weight change
  6. "worthlessness" - feeling bad about yourself, worthless, or excessively guilty
  7. "concentration" - trouble concentrating on things
  8. "psychomotor" - moving or speaking noticeably slowly, or being restless and fidgety
  9. "self_harm" - thoughts that you would be better off dead, or of hurting yourself

For each symptom return one of:
  "present" - the post gives positive evidence that the writer experiences it
  "absent"  - the post gives positive evidence that the writer does NOT experience it
  "unclear" - the post does not say either way

Use "unclear" when the post is simply silent about a symptom. Do not use "absent" merely because a symptom is unmentioned.

When and only when a symptom is "present", also return "evidence": a short span copied verbatim from the post. Copy it exactly; do not paraphrase.

Reply with JSON only - no preamble, no code fence, no commentary. Schema:
{"<symptom>": {"status": "present|absent|unclear", "evidence": "<verbatim span or empty>"}, ...}
Include all nine symptom keys exactly as written above.
```

### user

```
Post:
"""
{text}
"""
```

## C4

as C3 over BDI-II's 21 items.

### system

```
You are annotating a single social media post for the presence of twenty-one specific symptoms. You are not rating, scoring, or diagnosing anything - you only report, for each symptom, whether the post gives evidence of it.

The twenty-one symptoms are:
  1. "sadness" - feeling sad or unhappy
  2. "pessimism" - feeling discouraged or hopeless about the future
  3. "past_failure" - feeling like a failure, or dwelling on past failures
  4. "loss_of_pleasure" - getting less pleasure from things previously enjoyed
  5. "guilt" - feeling guilty
  6. "punishment" - feeling one is being punished, or deserves punishment
  7. "self_dislike" - disliking oneself, or having lost confidence in oneself
  8. "self_criticism" - blaming or criticising oneself
  9. "suicidal" - thoughts of killing oneself, or of being better off dead
  10. "crying" - crying, or being unable to cry when one wants to
  11. "agitation" - feeling restless, agitated, or keyed up
  12. "loss_of_interest" - having lost interest in other people or activities
  13. "indecisiveness" - finding it harder than usual to make decisions
  14. "worthlessness" - feeling worthless, or of no value
  15. "loss_of_energy" - having less energy than usual
  16. "sleep_change" - sleeping more or less than usual, or broken sleep
  17. "irritability" - being more irritable than usual
  18. "appetite_change" - eating more or less than usual, or appetite change
  19. "concentration" - finding it harder than usual to concentrate
  20. "fatigue" - being too tired to do many of the things one used to do
  21. "loss_of_interest_sex" - reduced interest in sex

For each symptom return one of:
  "present" - the post gives positive evidence that the writer experiences it
  "absent"  - the post gives positive evidence that the writer does NOT experience it
  "unclear" - the post does not say either way

Use "unclear" when the post is simply silent about a symptom. Do not use "absent" merely because a symptom is unmentioned.

When and only when a symptom is "present", also return "evidence": a short span copied verbatim from the post. Copy it exactly; do not paraphrase.

Reply with JSON only - no preamble, no code fence, no commentary. Schema:
{"<symptom>": {"status": "present|absent|unclear", "evidence": "<verbatim span or empty>"}, ...}
Include all twenty-one symptom keys exactly as written above.
```

### user

```
Post:
"""
{text}
"""
```

## C3P

**prompt ablation** — C3 with a lower evidentiary bar for `present`. Extracts 60% more symptoms and scores no better.

### system

```
You are annotating a single social media post for the presence of nine specific symptoms. You are not rating, scoring, or diagnosing anything - you only report, for each symptom, whether the post indicates it.

The nine symptoms are:
  1. "anhedonia" - little interest or pleasure in doing things
  2. "depressed_mood" - feeling down, depressed, or hopeless
  3. "sleep" - trouble falling or staying asleep, or sleeping too much
  4. "fatigue" - feeling tired or having little energy
  5. "appetite" - poor appetite, overeating, or weight change
  6. "worthlessness" - feeling bad about yourself, worthless, or excessively guilty
  7. "concentration" - trouble concentrating on things
  8. "psychomotor" - moving or speaking noticeably slowly, or being restless and fidgety
  9. "self_harm" - thoughts that you would be better off dead, or of hurting yourself

For each symptom return one of:
  "present" - the post indicates the writer experiences it. Count it as present if the writer states it directly, describes it in their own words, or describes circumstances or behaviour from which it reasonably follows. Do not require a clinical phrasing, and do not require the writer to name the symptom.
  "absent"  - the post indicates the writer does NOT experience it
  "unclear" - the post gives no indication either way

Err toward "present" when a reading of the post supports it. Reserve "unclear" for symptoms the post genuinely does not touch on.

When a symptom is "present", also return "evidence": a short span copied verbatim from the post that supports the judgement. Copy it exactly; do not paraphrase.

Reply with JSON only - no preamble, no code fence, no commentary. Schema:
{"<symptom>": {"status": "present|absent|unclear", "evidence": "<verbatim span or empty>"}, ...}
Include all nine symptom keys exactly as written above.
```

### user

```
Post:
"""
{text}
"""
```

## C3S

**prompt ablation** — C3 requiring a verbatim span for `absent` as well as `present`, testing whether the three-valued collapse is an artifact of the wording.

### system

```
You are annotating a single social media post for the presence of nine specific symptoms. You are not rating, scoring, or diagnosing anything - you only report, for each symptom, what the post says about it.

The nine symptoms are:
  1. "anhedonia" - little interest or pleasure in doing things
  2. "depressed_mood" - feeling down, depressed, or hopeless
  3. "sleep" - trouble falling or staying asleep, or sleeping too much
  4. "fatigue" - feeling tired or having little energy
  5. "appetite" - poor appetite, overeating, or weight change
  6. "worthlessness" - feeling bad about yourself, worthless, or excessively guilty
  7. "concentration" - trouble concentrating on things
  8. "psychomotor" - moving or speaking noticeably slowly, or being restless and fidgety
  9. "self_harm" - thoughts that you would be better off dead, or of hurting yourself

For each symptom return one of:
  "present" - the post gives evidence that the writer experiences it
  "absent"  - the post gives evidence that the writer does NOT experience it. This includes the writer denying the symptom, describing its opposite, or describing functioning that is incompatible with it.
  "unclear" - the post does not address the symptom

For BOTH "present" and "absent" you must return "evidence": a short span copied verbatim from the post that supports the judgement. If you cannot quote a supporting span, the correct answer is "unclear". Copy spans exactly; do not paraphrase.

Reply with JSON only - no preamble, no code fence, no commentary. Schema:
{"<symptom>": {"status": "present|absent|unclear", "evidence": "<verbatim span or empty>"}, ...}
Include all nine symptom keys exactly as written above.
```

### user

```
Post:
"""
{text}
"""
```