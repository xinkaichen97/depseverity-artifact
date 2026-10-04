# C3 capped to C2's resolution, against recalibrated C2

C3's count is capped at k-1 and relabeled by the best monotone map on the fitting
split, the same search C2+cal gets (calibrate.py). Paired bootstrap, 4000 resamples.

| Corpus | Model | C3 capped map | C3 fitted $-$ C2+cal | sig | C3 capped $-$ C2+cal | sig |
|---|---|---|---|:--:|---|:--:|
| depseverity | Qwen3.5-9B | `0233` | +0.109 [+0.027, +0.189] | **yes** | +0.109 [+0.027, +0.189] | **yes** |
| depseverity | DeepSeek-V4.1-Flash | `0123` | +0.022 [-0.039, +0.086] | no | +0.022 [-0.039, +0.086] | no |
| depseverity | Claude-Sonnet-5 | `0123` | +0.037 [-0.029, +0.103] | no | +0.037 [-0.029, +0.103] | no |
| depsign | Qwen3.5-9B | `011` | +0.045 [-0.024, +0.112] | no | +0.032 [-0.034, +0.097] | no |
| depsign | DeepSeek-V4.1-Flash | `011` | -0.059 [-0.119, +0.003] | no | -0.072 [-0.134, -0.011] | **yes** |
| depsign | Claude-Sonnet-5 | `011` | -0.028 [-0.083, +0.028] | no | -0.040 [-0.104, +0.026] | no |