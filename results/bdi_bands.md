# C4 under BDI-II's own bands

BDI-II's bands are defined on its native 0-63 sum of graded items. Rescaled onto
our 0-21 binary present count (aggregate.py), they give the cutoffs below. C4 is
compared with C2 on the same posts.

| Corpus | Model | cutoffs | C4 QWK | dQWK vs C2 [95% CI] | sig |
|---|---|---|---:|---|:--:|
| depseverity | DeepSeek-V4.1-Flash | [4.5, 6.5, 9.5] | 0.079 | -0.425 [-0.504, -0.344] | **yes** |
| depseverity | Claude-Sonnet-5 | [4.5, 6.5, 9.5] | 0.110 | -0.352 [-0.431, -0.269] | **yes** |
| depsign | DeepSeek-V4.1-Flash | [4.5, 9.5] | 0.119 | -0.109 [-0.167, -0.054] | **yes** |
| depsign | Claude-Sonnet-5 | [4.5, 9.5] | 0.152 | -0.068 [-0.130, -0.007] | **yes** |