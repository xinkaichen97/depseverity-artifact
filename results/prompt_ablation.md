# Prompt ablation --- is the null an artifact of the extraction prompt?

DeepSeek-V4.1-Flash on DepSeverity. Same nine items, same JSON schema, same prohibition
on the model seeing or emitting a severity label; only the wording changes.

| Variant | present/post | absent | grounded | $\kappa_w$ fitted | $\kappa_w$ a priori |
|---|---:|---:|---:|---:|---:|
| original | 0.45 | 0.30% | 100.0% | 0.526 | 0.423 |
| permissive | 0.72 | 0.25% | 100.0% | 0.507 | 0.425 |
| symmetric evidence | 0.47 | 0.35% | 100.0% | 0.555 | 0.433 |

| Variant | vs C2, fitted | vs C2, a priori |
|---|---|---|
| original | $+0.022$ [-0.039, +0.086] ns | $-0.081$ [-0.147, -0.010] **sig** |
| permissive | $+0.002$ [-0.059, +0.067] ns | $-0.079$ [-0.145, -0.012] **sig** |
| symmetric evidence | $+0.051$ [-0.011, +0.116] ns | $-0.071$ [-0.138, +0.000] ns |