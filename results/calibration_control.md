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

## Other extraction variants against a supervision-matched C1/C2 (paired, 4000 resamples)

| Corpus | Model | Comparison | $\Delta\kappa_w$ [95\% CI] | sig |
|---|---|---|---|:--:|
| depseverity | qwen3.5 | C3 a priori $-$ C2+cal | $-0.058$ [-0.136, +0.023] | no |
| depseverity | qwen3.5 | C4 fitted $-$ C2+cal | $+0.116$ [+0.024, +0.202] | **yes** |
| depseverity | qwen3.5 | C4 a priori $-$ C2+cal | $+0.091$ [-0.001, +0.181] | no |
| depseverity | qwen3.5 | C3 a priori $-$ C1+cal | $+0.002$ [-0.072, +0.079] | no |
| depseverity | qwen3.5 | C4 fitted $-$ C1+cal | $+0.176$ [+0.085, +0.259] | **yes** |
| depseverity | qwen3.5 | C4 a priori $-$ C1+cal | $+0.150$ [+0.060, +0.235] | **yes** |
| depseverity | deepseek-fla | C3 a priori $-$ C2+cal | $-0.081$ [-0.147, -0.010] | **yes** |
| depseverity | deepseek-fla | C4 fitted $-$ C2+cal | $-0.035$ [-0.107, +0.039] | no |
| depseverity | deepseek-fla | C4 a priori $-$ C2+cal | $-0.026$ [-0.100, +0.049] | no |
| depseverity | deepseek-fla | C3 a priori $-$ C1+cal | $+0.094$ [+0.013, +0.173] | **yes** |
| depseverity | deepseek-fla | C4 fitted $-$ C1+cal | $+0.139$ [+0.050, +0.226] | **yes** |
| depseverity | deepseek-fla | C4 a priori $-$ C1+cal | $+0.149$ [+0.061, +0.236] | **yes** |
| depseverity | claude-sonne | C3 a priori $-$ C2+cal | $-0.067$ [-0.135, +0.005] | no |
| depseverity | claude-sonne | C4 fitted $-$ C2+cal | $-0.002$ [-0.075, +0.072] | no |
| depseverity | claude-sonne | C4 a priori $-$ C2+cal | $+0.004$ [-0.068, +0.079] | no |
| depseverity | claude-sonne | C3 a priori $-$ C1+cal | $+0.043$ [-0.039, +0.125] | no |
| depseverity | claude-sonne | C4 fitted $-$ C1+cal | $+0.107$ [+0.023, +0.190] | **yes** |
| depseverity | claude-sonne | C4 a priori $-$ C1+cal | $+0.114$ [+0.030, +0.199] | **yes** |
| depsign | qwen3.5 | C3 a priori $-$ C2+cal | $+0.036$ [-0.030, +0.101] | no |
| depsign | qwen3.5 | C4 fitted $-$ C2+cal | $+0.016$ [-0.049, +0.083] | no |
| depsign | qwen3.5 | C4 a priori $-$ C2+cal | $+0.001$ [-0.061, +0.064] | no |
| depsign | qwen3.5 | C3 a priori $-$ C1+cal | $+0.069$ [+0.003, +0.138] | **yes** |
| depsign | qwen3.5 | C4 fitted $-$ C1+cal | $+0.050$ [-0.019, +0.119] | no |
| depsign | qwen3.5 | C4 a priori $-$ C1+cal | $+0.034$ [-0.031, +0.102] | no |
| depsign | deepseek-fla | C3 a priori $-$ C2+cal | $-0.055$ [-0.120, +0.010] | no |
| depsign | deepseek-fla | C4 fitted $-$ C2+cal | $-0.044$ [-0.120, +0.027] | no |
| depsign | deepseek-fla | C4 a priori $-$ C2+cal | $-0.084$ [-0.151, -0.018] | **yes** |
| depsign | deepseek-fla | C3 a priori $-$ C1+cal | $-0.039$ [-0.112, +0.035] | no |
| depsign | deepseek-fla | C4 fitted $-$ C1+cal | $-0.028$ [-0.098, +0.042] | no |
| depsign | deepseek-fla | C4 a priori $-$ C1+cal | $-0.068$ [-0.130, -0.003] | **yes** |
| depsign | claude-sonne | C3 a priori $-$ C2+cal | $-0.013$ [-0.074, +0.052] | no |
| depsign | claude-sonne | C4 fitted $-$ C2+cal | $-0.041$ [-0.110, +0.024] | no |
| depsign | claude-sonne | C4 a priori $-$ C2+cal | $-0.062$ [-0.113, -0.011] | **yes** |
| depsign | claude-sonne | C3 a priori $-$ C1+cal | $-0.027$ [-0.106, +0.052] | no |
| depsign | claude-sonne | C4 fitted $-$ C1+cal | $-0.056$ [-0.132, +0.021] | no |
| depsign | claude-sonne | C4 a priori $-$ C1+cal | $-0.077$ [-0.139, -0.012] | **yes** |