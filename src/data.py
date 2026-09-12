"""Load, clean and split the DepSeverity (Naseem et al., WWW'22) dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data/raw/Depression_Severity_Dataset/Reddit_depression_dataset.csv"
DREADDIT = ROOT / "data/raw/dreaddit"
PROCESSED = ROOT / "data/processed"

LABELS = ["minimum", "mild", "moderate", "severe"]  # ordinal, low -> high
LABEL2ORD = {lab: i for i, lab in enumerate(LABELS)}
SEED = 20260909


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    df["label"] = df["label"].str.strip().str.lower()
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop label-conflicting duplicates, collapse exact duplicates.

    Exact duplicate texts must not straddle the train/test boundary (leakage).
    Texts carrying two different gold labels are unresolvable, so both copies go.
    """
    n0 = len(df)
    counts = df.groupby("text")["label"].nunique()
    conflicted = set(counts[counts > 1].index)
    df = df[~df["text"].isin(conflicted)].copy()
    n_conflict_rows = n0 - len(df)

    n1 = len(df)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    n_dup_rows = n1 - len(df)

    df["y"] = df["label"].map(LABEL2ORD)
    stats = {
        "rows_raw": n0,
        "conflicting_texts": len(conflicted),
        "rows_dropped_conflict": n_conflict_rows,
        "rows_dropped_duplicate": n_dup_rows,
        "rows_clean": len(df),
    }
    return df, stats


def link_dreaddit(df: pd.DataFrame) -> pd.DataFrame:
    """Recover source subreddit + Dreaddit stress label by exact text match.

    DepSeverity is Dreaddit (Turcan & McKeown 2019) relabelled for depression
    severity: same 3,553 posts, 100% exact text match. Dreaddit carries the
    source subreddit, which DepSeverity drops. We need it for error analysis.
    """
    dr = pd.concat(
        [pd.read_csv(DREADDIT / f"dreaddit-{s}.csv").assign(dreaddit_split=s)
         for s in ("train", "test")],
        ignore_index=True,
    )
    key = lambda s: s.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    dr = dr.assign(k=key(dr["text"])).drop_duplicates("k")
    cols = ["k", "subreddit", "dreaddit_split", "label", "confidence"]
    out = df.assign(k=key(df["text"])).merge(
        dr[cols].rename(columns={"label": "stress", "confidence": "stress_conf"}),
        on="k", how="left",
    ).drop(columns="k")
    out.index = df.index
    return out


def make_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED) -> pd.DataFrame:
    tr, te = train_test_split(
        df.index, test_size=test_size, random_state=seed, stratify=df["y"]
    )
    df = df.copy()
    df["split"] = "train"
    df.loc[te, "split"] = "test"
    return df


def make_dev_subset(df: pd.DataFrame, per_class: int = 8, seed: int = SEED) -> pd.Index:
    """Prompt-development subset, drawn from TRAIN only.

    Class-balanced, not stratified: a stratified draw of 30 gives only 2 severe
    posts, and prompt iteration needs coverage of the adjacent-level boundaries.
    It is never scored, so balance costs nothing.
    """
    train = df[df["split"] == "train"]
    idx = [
        train[train["label"] == lab].sample(per_class, random_state=seed).index
        for lab in LABELS
    ]
    return pd.Index(np.concatenate(idx))


def make_fit_subset(df: pd.DataFrame, n: int = 600, seed: int = SEED + 1) -> pd.Index:
    """Stratified train posts used to fit the three aggregation cutoffs.

    Three thresholds over an integer 0-9 score need far fewer than the full 2,824
    train posts, and C3 extraction on the local model costs ~11s per post. 600 is
    ample for three parameters; the choice is frozen here so it is reproducible.
    Drawn from train, excluding the prompt-development posts.
    """
    pool = df[(df["split"] == "train") & (~df["is_dev"])]
    _, fit = train_test_split(pool.index, test_size=n, random_state=seed,
                              stratify=pool.loc[pool.index, "y"])
    return fit


def build(save: bool = True) -> pd.DataFrame:
    df, stats = clean(load_raw())
    df = link_dreaddit(df)
    stats["dreaddit_matched"] = int(df["subreddit"].notna().sum())
    df = make_split(df)
    dev = make_dev_subset(df)
    df["is_dev"] = df.index.isin(dev)
    df["is_fit"] = df.index.isin(make_fit_subset(df))
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED / "depseverity_split.csv", index_label="id")
    return df, stats


def load() -> pd.DataFrame:
    """Load the frozen split. Every downstream run must use this."""
    return pd.read_csv(PROCESSED / "depseverity_split.csv", index_col="id")


if __name__ == "__main__":
    df, stats = build()
    print(stats)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
