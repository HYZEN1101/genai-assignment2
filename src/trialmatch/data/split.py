"""Patient-grouped 37/8/8 split with seed 42 (assignment protocol).

Protocol: sorted unique patient IDs (plain Python string sort) -> numpy array -> np.random.default_rng(42).shuffle(array)
(in place) -> first 37 = train, next 8 = val, last 8 = test. All rows of a patient go to that patient's split.
"""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np
import pandas as pd

SEED = 42
SIZES = {"train": 37, "val": 8, "test": 8}
SPLITS = tuple(SIZES)
DATASET = "ncbi/TrialGPT-Criterion-Annotations"
METHOD = (
    "ids = sorted(set(patient_id)) [python str sort]; arr = np.array(ids); "
    "rng = np.random.default_rng(42); rng.shuffle(arr) [in place]; "
    "train = arr[:37], val = arr[37:45], test = arr[45:53]"
)


def split_patient_ids(patient_ids: Sequence[str], seed: int = SEED, sizes: dict[str, int] = SIZES) -> dict[str, list[str]]:
    ids = sorted(set(str(p) for p in patient_ids))
    if len(ids) != sum(sizes.values()):
        raise ValueError(f"expected {sum(sizes.values())} unique patients, found {len(ids)}")
    arr = np.array(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    out, start = {}, 0
    for name, n in sizes.items():
        out[name] = [str(x) for x in arr[start:start + n]]
        start += n
    return out


def rows_fingerprint(df: pd.DataFrame) -> str:
    """Hash of (annotation_id, patient_id) pairs: ties the manifest to the exact raw rows it was computed from."""
    pairs = sorted(f"{int(a)}|{p}" for a, p in zip(df["annotation_id"], df["patient_id"]))
    return hashlib.sha256("\n".join(pairs).encode()).hexdigest()


def build_manifest(df: pd.DataFrame) -> dict:
    parts = split_patient_ids(df["patient_id"].tolist())
    manifest = {
        "dataset": DATASET,
        "seed": SEED,
        "method": METHOD,
        "n_rows_total": int(len(df)),
        "n_patients_total": int(df["patient_id"].nunique()),
        "rows_fingerprint_sha256": rows_fingerprint(df),
        "splits": {},
    }
    for name, pids in parts.items():
        mask = df["patient_id"].isin(pids)
        manifest["splits"][name] = {"n_patients": len(pids), "n_rows": int(mask.sum()), "patient_ids": pids}
    canon = json.dumps({k: v["patient_ids"] for k, v in manifest["splits"].items()}, sort_keys=True)
    manifest["patient_assignment_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    return manifest


def manifest_to_json(manifest: dict) -> str:
    """Deterministic serialisation (no timestamps) so regeneration is byte-identical."""
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def verify_manifest(manifest: dict, df: pd.DataFrame) -> None:
    """Raise AssertionError if the manifest is inconsistent with df (leakage / count / fingerprint checks)."""
    sp = manifest["splits"]
    sets = {k: set(v["patient_ids"]) for k, v in sp.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b]), f"patient leakage between {a} and {b}: {sorted(sets[a] & sets[b])}"
    for k, v in sp.items():
        assert len(v["patient_ids"]) == len(sets[k]) == v["n_patients"] == SIZES[k], f"bad patient count for {k}"
    assert set().union(*sets.values()) == set(df["patient_id"].astype(str)), "manifest patients != dataset patients"
    assert manifest["rows_fingerprint_sha256"] == rows_fingerprint(df), "raw data changed since manifest was made"
    total = 0
    for k, v in sp.items():
        n = int(df["patient_id"].isin(sets[k]).sum())
        assert n == v["n_rows"], f"row count mismatch for {k}: {n} vs {v['n_rows']}"
        total += n
    assert total == len(df) == manifest["n_rows_total"], "rows lost or duplicated across splits"


def assign_split(df: pd.DataFrame, manifest: dict) -> pd.Series:
    lookup = {p: k for k, v in manifest["splits"].items() for p in v["patient_ids"]}
    return df["patient_id"].astype(str).map(lookup)
