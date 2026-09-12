# Second corpus — DepSign (pre-registered replication)

Identical prompts, harness, cutoff protocol and models; only the corpus changes.
706-post stratified subsample of DepSign's **official test split** (50 severe);
cutoffs fitted on 600 posts from its **official train split**. Three classes, so
two cutoffs and the a priori rule is DSM-5 >=5 of 9 -> severe, 0 -> not depression.

| Model | Condition | cutoffs | QWK [95% CI] | MAE | Acc | Top-class missed |
|---|---|---|---|---:|---:|---|
| V4.1-Flash | C1 (direct) | — | **0.109** [0.073, 0.145] | 0.956 | 0.184 | 9/50 |
| V4.1-Flash | C2 (CoT) | — | **0.228** [0.172, 0.280] | 0.737 | 0.344 | 16/50 |
| V4.1-Flash | C3 fitted | [0.5, 5.5] | **0.203** [0.133, 0.269] | 0.347 | 0.660 | 49/50 |
| V4.1-Flash | C3 a priori | [0.5, 4.5] | **0.207** [0.134, 0.273] | 0.407 | 0.613 | 42/50 |
| sonnet-5 | C1 (direct) | — | **0.151** [0.112, 0.189] | 0.796 | 0.288 | 17/50 |
| sonnet-5 | C2 (CoT) | — | **0.220** [0.167, 0.273] | 0.705 | 0.364 | 19/50 |
| sonnet-5 | C3 fitted | [0.5, 3.5] | **0.192** [0.127, 0.256] | 0.498 | 0.536 | 36/50 |
| sonnet-5 | C3 a priori | [0.5, 4.5] | **0.207** [0.138, 0.278] | 0.389 | 0.626 | 42/50 |

## Primary pre-registered comparison (paired, 4000 resamples)

| Model | Comparison | dQWK [95% CI] | sig | dMAE |
|---|---|---|:--:|---|
| V4.1-Flash | C2 − C1 | +0.119 [+0.078, +0.160] | **yes** | -0.220 [-0.266, -0.174] sig |
| V4.1-Flash | C3 fitted − C2 | -0.026 [-0.086, +0.033] | no | -0.390 [-0.439, -0.339] sig |
| V4.1-Flash | C3 a priori − C2 | -0.022 [-0.080, +0.036] | no | -0.330 [-0.378, -0.282] sig |
| sonnet-5 | C2 − C1 | +0.069 [+0.036, +0.101] | **yes** | -0.091 [-0.126, -0.057] sig |
| sonnet-5 | C3 fitted − C2 | -0.028 [-0.084, +0.029] | no | -0.207 [-0.255, -0.162] sig |
| sonnet-5 | C3 a priori − C2 | -0.013 [-0.073, +0.051] | no | -0.316 [-0.365, -0.268] sig |