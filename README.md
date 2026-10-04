# Artifact for "Structure vs. Chain-of-Thought: Evaluating LLM Criteria Extraction for Depression Severity"

Paper: [arXiv:2609.39049](https://arxiv.org/abs/2609.39049) (extended version; accepted at MHSM 2026, an IEEE ICDM 2026 workshop).

Code, exact prompts, aggregate results, and scrubbed per-post records for the paper.

## What is here

- `src/` — the full pipeline: providers, conditions, aggregation, metrics, analyses.
- `PROMPTS.md` — the exact system and user prompts for every condition,
  C1–C4 plus the C3P/C3S prompt ablation, verbatim.
- `runs_scrubbed/` — one record per (condition, model, corpus, split, seed) post.
  **No post text.** Raw model responses and verbatim evidence spans are removed;
  per-item `status` values, gold labels, predictions and token counts are retained,
  which is everything the reported metrics are computed from.
- `results/` — the generated tables and JSON behind every number in the paper.
- `paper/` — `numbers.json` and `macro_f1_deltas.json`, which every number in the paper is
  read from, plus `fig_deltas.py` (Fig. 1) `macro_f1_boot.py` (macro-F1 paired
  bootstrap, which runs on `runs_scrubbed/` and first checks it reproduces `numbers.json`) and
  `run_to_run.py` (generation variance from the seed-1 repeats of Claude-Sonnet-5 and
  DeepSeek-V4.1-Flash). For the extended version's appendix: `per_class_recall.py`
  (per-class recall for every cell, checked against `numbers.json` first) and
  `full_appendix.py` (writes the appendix tables and prompts as LaTeX), `fig_tradeoff.py`
  (agreement vs. missed SEVERE posts) and `fig_counts.py` (criteria marked per post).

## What is deliberately absent

**No corpus text and no response cache.** Neither corpus is distributed with a stated
license, so we redistribute neither, and the response cache embeds the prompts and
therefore the posts. Obtain the corpora from their own sources, then re-run the
pipeline; all model calls are cached locally on first run.

## Reproducing

```bash
pip install -r requirements.txt
python src/data.py                      # build + freeze the DepSeverity split
python src/data_depsign.py              # build + freeze the DepSign subsamples
python src/run.py --condition C3 --model deepseek:deepseek-flash --subset test
python src/results.py                   # main tables
python src/depsign_results.py           # second-corpus tables
python src/control_cutoffs.py           # fitted vs a priori control
python src/bdi_bands.py                 # C4 under BDI-II's own bands
python src/stricter_thresholds.py       # C3 with stricter count thresholds (Sec. IV-B)
python src/ceiling.py                   # learned-aggregation ceiling
python src/error_analysis.py            # item 9, span grounding, per-community
python src/calibrate.py                 # supervision-matched C1/C2 control
python src/capped_count.py              # same control with C3 capped to C2's resolution (Sec. V-C)
python src/paper_numbers.py             # paper/numbers.json
cd paper && python fig_deltas.py && python macro_f1_boot.py   # Fig. 1, macro-F1 tests
cd paper && python per_class_recall.py && python full_appendix.py   # extended-version appendix
cd paper && python fig_tradeoff.py && python fig_counts.py   # extended-version figures
```

Every metric in the paper can be recomputed from `runs_scrubbed/` alone, without any
model calls, using the functions in `src/aggregate.py` and `src/evaluate.py`.

## Notes

- Reasoning is explicitly disabled per vendor; `meta.reasoning_leak` is non-zero if it
  was not, and every run in `runs_scrubbed/` reports 0.
- `meta.called_utc` records when each call was made. This matters: one hosted model ID
  was re-pointed to a different model during the study period.
- Scrubbing removed 32467 verbatim evidence spans across 49240 records.
