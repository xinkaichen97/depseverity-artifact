# Artifact — anonymous submission

Code, exact prompts, aggregate results, and scrubbed per-post records for the paper.

## What is here

- `src/` — the full pipeline: providers, conditions, aggregation, metrics, analyses.
- `PROMPTS.md` — the exact system and user prompts for C1–C4, verbatim.
- `runs_scrubbed/` — one record per (condition, model, corpus, split, seed) post.
  **No post text.** Raw model responses and verbatim evidence spans are removed;
  per-item `status` values, gold labels, predictions and token counts are retained,
  which is everything the reported metrics are computed from.
- `results/` — the generated tables and JSON behind every number in the paper.

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
```

Every metric in the paper can be recomputed from `runs_scrubbed/` alone, without any
model calls, using the functions in `src/aggregate.py` and `src/evaluate.py`.

## Notes

- Reasoning is explicitly disabled per vendor; `meta.reasoning_leak` is non-zero if it
  was not, and every run in `runs_scrubbed/` reports 0.
- `meta.called_utc` records when each call was made. This matters: one hosted model ID
  was re-pointed to a different model during the study period.
- Scrubbing removed 24883 verbatim evidence spans across 33462 records.
