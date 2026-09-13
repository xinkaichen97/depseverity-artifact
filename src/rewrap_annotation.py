"""Rewrap the post text in the annotation CSVs for readable editing.

CSV quoted fields may legally contain newlines, so this stays a valid CSV that pandas
round-trips. Only the display of `text` changes; scoring joins on `id` and reads gold
from the frozen split, so the wrapping cannot affect any result.
"""
import sys
import textwrap
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 88


def wrap(t: str) -> str:
    t = str(t).replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for para in t.split("\n"):
        para = para.strip()
        out.append("\n".join(textwrap.wrap(para, WIDTH)) if para else "")
    return "\n".join(out).strip()


def main(ds):
    p = ROOT / f"notes/annotation_blind_{ds}.csv"
    df = pd.read_csv(p, index_col="id")
    before = (len(df), df["my_label"].notna().sum())
    # Start each post on its own line: the editable fields then sit alone on the
    # record's first line and the body reads as an indented-looking block. A leading
    # newline inside a quoted CSV field is legal and round-trips. Display-only.
    df["text"] = df["text"].map(lambda t: "\n" + wrap(t))
    df.to_csv(p, index_label="id")

    back = pd.read_csv(p, index_col="id")          # round-trip check
    assert len(back) == before[0], "row count changed"
    assert list(back.index) == list(df.index), "ids changed"
    assert back["my_label"].fillna("").tolist() == df["my_label"].fillna("").tolist(), \
        "labels changed"
    done = int((back["my_label"].astype(str).str.strip() != "").sum() -
               (back["my_label"].isna()).sum() * 0)
    print(f"  {ds}: {len(back)} rows, {done} already labeled, round-trip OK "
          f"(wrapped at {WIDTH} cols)")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["depseverity", "depsign"]):
        main(ds)
