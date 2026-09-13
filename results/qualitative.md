# Worked examples --- deepseek:deepseek-flash, DepSeverity test split

No post text: neither corpus is licensed, so cases are identified by split
`id` for readers who hold the data. Fitted cutoffs [0.5, 1.5, 2.5]; *a priori* [0.5, 2.5, 4.5].

### severe missed at the aggregation floor --- post `8` (r/relationships)
_gold severe, but too few items fire for any cutoff to reach severe_

- gold **severe** | C2 minimum | C3 fitted **mild** | C3 *a priori* **mild**
- score 1 of 9; present: depressed_mood*
- unclear: anhedonia, sleep, fatigue, appetite, worthlessness, concentration, psychomotor, self_harm (`*` = evidence span returned)

### severe missed at the aggregation floor --- post `26` (r/anxiety)
_gold severe, but too few items fire for any cutoff to reach severe_

- gold **severe** | C2 mild | C3 fitted **minimum** | C3 *a priori* **minimum**
- score 0 of 9; present: none
- unclear: anhedonia, depressed_mood, sleep, fatigue, appetite, worthlessness, concentration, psychomotor, self_harm (`*` = evidence span returned)

### threshold regime flips the label --- post `56` (r/ptsd)
_same extraction; fitted and a priori cutoffs disagree_

- gold **minimum** | C2 minimum | C3 fitted **moderate** | C3 *a priori* **mild**
- score 2 of 9; present: depressed_mood*, psychomotor*
- unclear: anhedonia, sleep, fatigue, appetite, worthlessness, concentration, self_harm (`*` = evidence span returned)

### threshold regime flips the label --- post `90` (r/anxiety)
_same extraction; fitted and a priori cutoffs disagree_

- gold **severe** | C2 moderate | C3 fitted **moderate** | C3 *a priori* **mild**
- score 2 of 9; present: worthlessness*, concentration*
- unclear: anhedonia, depressed_mood, sleep, fatigue, appetite, psychomotor, self_harm (`*` = evidence span returned)

### threshold regime flips the label --- post `106` (r/anxiety)
_same extraction; fitted and a priori cutoffs disagree_

- gold **mild** | C2 moderate | C3 fitted **severe** | C3 *a priori* **moderate**
- score 3 of 9; present: anhedonia*, depressed_mood*, worthlessness*
- unclear: sleep, fatigue, appetite, concentration, psychomotor, self_harm (`*` = evidence span returned)

### chain-of-thought right, structure wrong --- post `29` (r/anxiety)
_C2 matches gold, C3 does not, on identical input_

- gold **moderate** | C2 moderate | C3 fitted **mild** | C3 *a priori* **mild**
- score 1 of 9; present: worthlessness*
- unclear: anhedonia, depressed_mood, sleep, fatigue, appetite, concentration, psychomotor, self_harm (`*` = evidence span returned)

### chain-of-thought right, structure wrong --- post `56` (r/ptsd)
_C2 matches gold, C3 does not, on identical input_

- gold **minimum** | C2 minimum | C3 fitted **moderate** | C3 *a priori* **mild**
- score 2 of 9; present: depressed_mood*, psychomotor*
- unclear: anhedonia, sleep, fatigue, appetite, worthlessness, concentration, self_harm (`*` = evidence span returned)

### item 9 as a standalone flag --- post `327` (r/survivorsofabuse)
_self-harm ideation fires on a gold-severe post_

- gold **severe** | C2 severe | C3 fitted **moderate** | C3 *a priori* **mild**
- score 2 of 9; present: depressed_mood*, self_harm*
- unclear: anhedonia, sleep, fatigue, appetite, worthlessness, concentration, psychomotor (`*` = evidence span returned)

### item 9 as a standalone flag --- post `649` (r/ptsd)
_self-harm ideation fires on a gold-severe post_

- gold **severe** | C2 severe | C3 fitted **severe** | C3 *a priori* **moderate**
- score 3 of 9; present: depressed_mood*, psychomotor*, self_harm*
- unclear: anhedonia, sleep, fatigue, appetite, worthlessness, concentration (`*` = evidence span returned)
