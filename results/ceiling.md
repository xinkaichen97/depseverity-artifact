# Ceiling — how much is in the nine items?

`count + fitted cutoffs` is the C3 rule (3 parameters). `learned 9-feature`
fits an ordinal model over the same nine extracted statuses on the same fit
split. It is an **upper bound** with many more parameters, not a fair rival to
C2 — it bounds what any aggregation of PHQ-9 items could achieve.

| Model | aggregation | QWK [95% CI] | MAE | Severe missed |
|---|---|---|---:|---|
| qwen3.5:9b | count + fitted cutoffs (C3) | **0.489** [0.410, 0.560] | 0.465 | 0.768 (43/56) |
| qwen3.5:9b | learned 9-feature, unweighted | **0.186** [0.111, 0.266] | 0.494 | 0.911 (51/56) |
| qwen3.5:9b | learned 9-feature, class-balanced | **0.454** [0.371, 0.532] | 0.482 | 0.464 (26/56) |
| qwen3.5:9b | _unweighted learned - C3_ | -0.303 [-0.380, -0.226] (sig) | | |
| DeepSeek-V4.1-Flash | count + fitted cutoffs (C3) | **0.526** [0.457, 0.592] | 0.452 | 0.875 (49/56) |
| DeepSeek-V4.1-Flash | learned 9-feature, unweighted | **0.156** [0.092, 0.227] | 0.501 | 0.982 (55/56) |
| DeepSeek-V4.1-Flash | learned 9-feature, class-balanced | **0.421** [0.347, 0.495] | 0.623 | 0.286 (16/56) |
| DeepSeek-V4.1-Flash | _unweighted learned - C3_ | -0.369 [-0.436, -0.301] (sig) | | |
| claude-sonnet-5 | count + fitted cutoffs (C3) | **0.499** [0.427, 0.568] | 0.467 | 0.875 (49/56) |
| claude-sonnet-5 | learned 9-feature, unweighted | **0.166** [0.095, 0.243] | 0.499 | 0.964 (54/56) |
| claude-sonnet-5 | learned 9-feature, class-balanced | **0.421** [0.344, 0.494] | 0.615 | 0.250 (14/56) |
| claude-sonnet-5 | _unweighted learned - C3_ | -0.333 [-0.403, -0.262] (sig) | | |

## Reference: C2 on the same test posts

| Model | QWK |
|---|---|
| qwen3.5:9b | 0.340 |
| DeepSeek-V4.1-Flash | 0.504 |
| claude-sonnet-5 | 0.462 |