# Control — fitted vs a priori cutoffs

C3's cutoffs are three parameters fitted on training labels; C1 and C2 get no
such affordance. If C3's advantage survives replacing them with an a priori
clinical rule (DSM-5 >=5 of 9 symptoms -> severe; cutoffs [0.5, 2.5, 4.5]),
the advantage is not an artefact of that asymmetry.

| Model | C3 variant | cutoffs | QWK [95% CI] | MAE | Acc | Severe missed |
|---|---|---|---|---:|---:|---|
| qwen3.5:9b | fitted | [0.5, 1, 1.5] | **0.489** [0.410, 0.560] | 0.465 | 0.728 | 0.768 (43/56) |
| qwen3.5:9b | a priori | [0.5, 2.5, 4.5] | **0.322** [0.262, 0.384] | 0.460 | 0.703 | 1.000 (56/56) |
| DeepSeek-V4.1-Flash | fitted | [0.5, 1.5, 2.5] | **0.526** [0.457, 0.592] | 0.452 | 0.670 | 0.875 (49/56) |
| DeepSeek-V4.1-Flash | a priori | [0.5, 2.5, 4.5] | **0.423** [0.366, 0.481] | 0.453 | 0.673 | 0.982 (55/56) |
| claude-sonnet-5 | fitted | [0.5, 1.5, 2.5] | **0.499** [0.427, 0.568] | 0.467 | 0.664 | 0.875 (49/56) |
| claude-sonnet-5 | a priori | [0.5, 2.5, 4.5] | **0.395** [0.331, 0.459] | 0.470 | 0.661 | 0.982 (55/56) |

## Does the C3 advantage over C2 survive? (paired, 4000 resamples)

| Model | Comparison | dQWK [95% CI] | sig |
|---|---|---|:--:|
| qwen3.5:9b | C3(fitted) - C2 | +0.149 [+0.072, +0.223] | **yes** |
| qwen3.5:9b | C3(a priori) - C2 | -0.018 [-0.089, +0.054] | no |
| DeepSeek-V4.1-Flash | C3(fitted) - C2 | +0.022 [-0.039, +0.086] | no |
| DeepSeek-V4.1-Flash | C3(a priori) - C2 | -0.081 [-0.147, -0.010] | **yes** |
| claude-sonnet-5 | C3(fitted) - C2 | +0.037 [-0.029, +0.103] | no |
| claude-sonnet-5 | C3(a priori) - C2 | -0.067 [-0.135, +0.005] | no |