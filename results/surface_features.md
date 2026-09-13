# Are the labels recoverable from non-criteria features?

Every feature is Dreaddit metadata or a surface property; none involves reading
the post for depressive symptoms. Fitted on train, evaluated on the frozen test
split, same ordinal construction and metrics as the LLM conditions.

| Feature set | $\kappa_w$ [95% CI] | MAE | Acc |
|---|---|---:|---:|
| source subreddit only (10 dummies) | **0.196** [0.146, 0.248] | 1.188 | 0.499 |
| post length only (2 features) | **0.011** [-0.044, 0.066] | 1.391 | 0.439 |
| Dreaddit stress label + confidence | **0.259** [0.212, 0.306] | 1.177 | 0.508 |
| subreddit + stress + length | **0.323** [0.265, 0.383] | 0.939 | 0.588 |
| + social signals (karma, votes, comments) | **0.343** [0.284, 0.400] | 0.902 | 0.601 |
| + LIWC/syntax lexicon counts | **0.404** [0.341, 0.467] | 0.745 | 0.650 |

Reference points on the same test split: best LLM condition $\kappa_w = 0.526$ (C3 fitted, DeepSeek-V4.1-Flash); chain-of-thought 0.462--0.504; TF-IDF text baseline 0.374; majority class 0.000.

## Paired comparison: metadata-only model vs each LLM cell

Positive $\Delta$ favors the LLM. Paired bootstrap on the posts both score.

| LLM cell | $\kappa_w$ | $\Delta$ vs metadata [95% CI] | $p$ |
|---|---:|---|---:|
| DeepSeek C2 (CoT) | 0.504 | +0.100 [+0.014, +0.183]* | 0.025 |
| DeepSeek C3, fitted | 0.526 | +0.122 [+0.040, +0.201]* | 0.003 |
| DeepSeek C3, a priori | 0.423 | +0.019 [-0.061, +0.100] | 0.652 |
| Claude-Sonnet-5 C2 (CoT) | 0.462 | +0.058 [-0.024, +0.138] | 0.170 |
| Claude-Sonnet-5 C3, fitted | 0.499 | +0.095 [+0.014, +0.176]* | 0.025 |
| Claude-Sonnet-5 C3, a priori | 0.395 | -0.009 [-0.089, +0.071] | 0.802 |

`*` = 95% CI excludes zero. Metadata-only model: $\kappa_w$ 0.404 (subreddit + stress + length + social + LIWC).