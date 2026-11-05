"""Load the raw TrialGPT Criterion Annotations table (cached as parquet in data/raw/, ~180 kB)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
REPO_ROOT = Path(__file__).resolve().parents[3]  # .../repo/src/trialmatch/data/load.py -> repo
RAW_CACHE = REPO_ROOT / "data" / "raw" / "trialgpt_criterion_annotations.parquet"


def load_raw(cache: Path = RAW_CACHE, refresh: bool = False) -> pd.DataFrame:
    """Return the full raw table (all columns, including GPT-4 ones: callers must select USED_COLUMNS)."""
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    from datasets import load_dataset  # lazy: tests need no network

    ds = load_dataset(DATASET_ID, split="train")
    df = ds.to_pandas()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df
