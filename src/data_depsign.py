"""DepSign (Sampath & Durairaj 2022, LT-EDI@ACL-2022) — the second corpus.

Same interface as data.py so the harness is unchanged: load() returns text, label,
y, split, is_fit. Three ordinal classes instead of four.

Parsing note: the released TSVs cannot be read with a plain TSV reader — posts
contain embedded tabs and newlines. Records are reconstructed on the PID prefix.
"""
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/depsign"
PROCESSED = ROOT / "data/processed"

LABELS = ["not depression", "moderate", "severe"]   # ordinal, low -> high
LABEL2ORD = {l: i for i, l in enumerate(LABELS)}
SEED = 20260911
N_TEST, N_FIT = 706, 600     # matched to DepSeverity so the corpora are comparable


def _read(split: str) -> pd.DataFrame:
    raw = (RAW / f"{split}.tsv").read_text(encoding="utf-8", errors="replace")
    parts = re.split(rf"(?m)^({split}_pid_\d+)\t", raw)[1:]
    rows = []
    for pid, body in zip(parts[0::2], parts[1::2]):
        *txt, lab = body.rstrip("\n").rsplit("\t", 1)
        rows.append({"pid": pid, "text": "\t".join(txt).strip(),
                     "label": lab.strip().lower()})
    df = pd.DataFrame(rows)
    return df[df["label"].isin(LABELS)].reset_index(drop=True)


def build(save: bool = True):
    tr, te = _read("train"), _read("test")
    stats = {"train_records": len(tr), "test_records": len(te)}

    # Stratified subsamples, frozen. Test comes from the official test split only.
    _, fit_idx = train_test_split(tr.index, test_size=N_FIT, random_state=SEED,
                                  stratify=tr["label"])
    _, test_idx = train_test_split(te.index, test_size=N_TEST, random_state=SEED,
                                   stratify=te["label"])
    fit = tr.loc[fit_idx].assign(split="fit")
    test = te.loc[test_idx].assign(split="test")

    df = pd.concat([fit, test], ignore_index=True)
    df["y"] = df["label"].map(LABEL2ORD)
    df["is_fit"] = df["split"] == "fit"
    df["is_dev"] = False
    stats |= {"fit": len(fit), "test": len(test),
              "test_dist": dict(test["label"].value_counts()),
              "fit_dist": dict(fit["label"].value_counts())}
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED / "depsign_split.csv", index_label="id")
    return df, stats


def load() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "depsign_split.csv", index_col="id")


if __name__ == "__main__":
    df, stats = build()
    for k, v in stats.items():
        print(f"  {k}: {v}")
