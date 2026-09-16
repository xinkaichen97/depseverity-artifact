# Symmetry control --- C1/C2 with the same labeled budget as C3

A monotone relabeling of the predicted class, fitted on the same 600-post
fitting split used for C3's thresholds, then frozen and applied to test.
Fewer free parameters than C3's threshold search.

| Corpus | Model | Cond. | raw $\kappa_w$ | +cal $\kappa_w$ | map | C3 fitted |
|---|---|---|---:|---:|---|---:|
| depseverity | qwen3.5 | C1 | 0.288 | **0.320** | `0012` | 0.489 |
| depseverity | qwen3.5 | C2 | 0.340 | **0.379** | `0013` | 0.489 |
| depseverity | deepseek-fla | C1 | 0.219 | **0.330** | `0002` | 0.526 |
| depseverity | deepseek-fla | C2 | 0.504 | **0.504** | `0123` | 0.526 |
| depseverity | claude-sonne | C1 | 0.299 | **0.352** | `0013` | 0.499 |
| depseverity | claude-sonne | C2 | 0.462 | **0.462** | `0123` | 0.499 |
| depsign | qwen3.5 | C1 | 0.162 | **0.185** | `011` | 0.264 |
| depsign | qwen3.5 | C2 | 0.112 | **0.219** | `001` | 0.264 |
| depsign | deepseek-fla | C1 | 0.109 | **0.246** | `001` | 0.203 |
| depsign | deepseek-fla | C2 | 0.228 | **0.262** | `011` | 0.203 |
| depsign | claude-sonne | C1 | 0.151 | **0.234** | `001` | 0.192 |
| depsign | claude-sonne | C2 | 0.220 | **0.220** | `012` | 0.192 |

## Does C3 still beat a supervision-matched C1/C2? (paired, 4000 resamples)

| Corpus | Model | Comparison | $\Delta\kappa_w$ [95\% CI] | sig |
|---|---|---|---|:--:|
| depseverity | qwen3.5 | C3 fitted $-$ C2+cal | $+0.109$ [+0.027, +0.189] | **yes** |
| depseverity | qwen3.5 | C3 fitted $-$ C1+cal | $+0.169$ [+0.090, +0.246] | **yes** |
| depseverity | deepseek-fla | C3 fitted $-$ C2+cal | $+0.022$ [-0.039, +0.086] | no |
| depseverity | deepseek-fla | C3 fitted $-$ C1+cal | $+0.196$ [+0.113, +0.277] | **yes** |
| depseverity | claude-sonne | C3 fitted $-$ C2+cal | $+0.037$ [-0.029, +0.103] | no |
| depseverity | claude-sonne | C3 fitted $-$ C1+cal | $+0.147$ [+0.067, +0.226] | **yes** |
| depsign | qwen3.5 | C3 fitted $-$ C2+cal | $+0.045$ [-0.024, +0.112] | no |
| depsign | qwen3.5 | C3 fitted $-$ C1+cal | $+0.078$ [+0.011, +0.148] | **yes** |
| depsign | deepseek-fla | C3 fitted $-$ C2+cal | $-0.059$ [-0.119, +0.003] | no |
| depsign | deepseek-fla | C3 fitted $-$ C1+cal | $-0.043$ [-0.116, +0.031] | no |
| depsign | claude-sonne | C3 fitted $-$ C2+cal | $-0.028$ [-0.083, +0.028] | no |
| depsign | claude-sonne | C3 fitted $-$ C1+cal | $-0.042$ [-0.112, +0.028] | no |