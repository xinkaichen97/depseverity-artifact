# Artifact — anonymous submission

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
  read from, plus `fig_deltas.py` (Fig. 1) and `macro_f1_boot.py` (macro-F1 paired
  bootstrap, which runs on `runs_scrubbed/` and first checks it reproduces `numbers.json`).

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
python src/ceiling.py                   # learned-aggregation ceiling
python src/error_analysis.py            # item 9, span grounding, per-community
python src/calibrate.py                 # supervision-matched C1/C2 control
python src/paper_numbers.py             # paper/numbers.json
cd paper && python fig_deltas.py && python macro_f1_boot.py   # Fig. 1, macro-F1 tests
```

Every metric in the paper can be recomputed from `runs_scrubbed/` alone, without any
model calls, using the functions in `src/aggregate.py` and `src/evaluate.py`.

## Notes

- Reasoning is explicitly disabled per vendor; `meta.reasoning_leak` is non-zero if it
  was not, and every run in `runs_scrubbed/` reports 0.
- `meta.called_utc` records when each call was made. This matters: one hosted model ID
  was re-pointed to a different model during the study period.
- Scrubbing removed 27156 verbatim evidence spans across 37380 records.
