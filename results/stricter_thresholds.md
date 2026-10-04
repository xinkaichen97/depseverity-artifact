# C3 with stricter count thresholds

SEVERE requires seven present criteria (bands 0-1 / 2-4 / 5-6 / 7+). On DepSign the
four bands collapse to three labels in two ways that keep that requirement. Each rule
is compared with C2 on the same posts (paired bootstrap, 4000 resamples).

| Corpus | Model | Rule | cutoffs | C3 QWK | SEVERE missed | dQWK vs C2 [95% CI] | sig |
|---|---|---|---|---:|---|---|:--:|
| depseverity | Qwen3.5-9B | standard a priori | [0.5, 2.5, 4.5] | 0.322 | 56/56 | -0.018 [-0.089, +0.054] | no |
| depseverity | Qwen3.5-9B | stricter | [1.5, 4.5, 6.5] | 0.117 | 56/56 | -0.223 [-0.288, -0.153] | **yes** |
| depseverity | DeepSeek-V4.1-Flash | standard a priori | [0.5, 2.5, 4.5] | 0.423 | 55/56 | -0.081 [-0.147, -0.010] | **yes** |
| depseverity | DeepSeek-V4.1-Flash | stricter | [1.5, 4.5, 6.5] | 0.231 | 56/56 | -0.273 [-0.334, -0.209] | **yes** |
| depseverity | Claude-Sonnet-5 | standard a priori | [0.5, 2.5, 4.5] | 0.395 | 55/56 | -0.067 [-0.135, +0.005] | no |
| depseverity | Claude-Sonnet-5 | stricter | [1.5, 4.5, 6.5] | 0.223 | 56/56 | -0.239 [-0.306, -0.170] | **yes** |
| depsign | Qwen3.5-9B | standard a priori | [0.5, 4.5] | 0.255 | 47/50 | +0.143 [+0.086, +0.198] | **yes** |
| depsign | Qwen3.5-9B | stricter, lowest two merged | [4.5, 6.5] | 0.023 | 50/50 | -0.089 [-0.128, -0.053] | **yes** |
| depsign | Qwen3.5-9B | stricter, middle two merged | [1.5, 6.5] | 0.204 | 50/50 | +0.092 [+0.034, +0.152] | **yes** |
| depsign | DeepSeek-V4.1-Flash | standard a priori | [0.5, 4.5] | 0.207 | 42/50 | -0.022 [-0.080, +0.036] | no |
| depsign | DeepSeek-V4.1-Flash | stricter, lowest two merged | [4.5, 6.5] | 0.045 | 50/50 | -0.183 [-0.238, -0.128] | **yes** |
| depsign | DeepSeek-V4.1-Flash | stricter, middle two merged | [1.5, 6.5] | 0.276 | 50/50 | +0.048 [-0.007, +0.101] | no |
| depsign | Claude-Sonnet-5 | standard a priori | [0.5, 4.5] | 0.207 | 42/50 | -0.013 [-0.074, +0.052] | no |
| depsign | Claude-Sonnet-5 | stricter, lowest two merged | [4.5, 6.5] | 0.039 | 50/50 | -0.181 [-0.234, -0.128] | **yes** |
| depsign | Claude-Sonnet-5 | stricter, middle two merged | [1.5, 6.5] | 0.253 | 50/50 | +0.033 [-0.027, +0.092] | no |