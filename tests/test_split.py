import json
import copy
import pytest
from trialmatch.data.load import REPO_ROOT, RAW_CACHE
from trialmatch.data.split import (SIZES, assign_split, build_manifest, manifest_to_json, split_patient_ids, verify_manifest)


def test_counts_and_disjoint(synth_df):
    parts = split_patient_ids(synth_df.patient_id)
    assert {k: len(v) for k, v in parts.items()} == SIZES
    allp = [p for v in parts.values() for p in v]
    assert len(allp) == len(set(allp)) == 53


def test_deterministic_and_order_independent(synth_df):
    a = split_patient_ids(synth_df.patient_id)
    b = split_patient_ids(synth_df.sample(frac=1, random_state=1).patient_id)
    assert a == b


def test_manifest_byte_identical_regeneration(synth_df):
    assert manifest_to_json(build_manifest(synth_df)) == manifest_to_json(build_manifest(synth_df.copy()))


def test_every_row_in_exactly_one_split(synth_df):
    m = build_manifest(synth_df)
    verify_manifest(m, synth_df)
    s = assign_split(synth_df, m)
    assert s.notna().all() and set(s) == {"train", "val", "test"}
    for pid, grp in synth_df.assign(s=s).groupby("patient_id"):
        assert grp.s.nunique() == 1  # all annotations of a patient together


def test_verify_detects_leakage(synth_df):
    m = build_manifest(synth_df)
    bad = copy.deepcopy(m)
    bad["splits"]["val"]["patient_ids"][0] = bad["splits"]["train"]["patient_ids"][0]
    with pytest.raises(AssertionError):
        verify_manifest(bad, synth_df)


def test_wrong_patient_count_raises(synth_df):
    with pytest.raises(ValueError):
        split_patient_ids(synth_df.patient_id.unique()[:50])


@pytest.mark.skipif(not (RAW_CACHE.exists() and (REPO_ROOT / "data" / "split_manifest.json").exists()), reason="real data not present")
def test_committed_manifest_matches_regeneration_on_real_data():
    import pandas as pd
    df = pd.read_parquet(RAW_CACHE)
    committed = (REPO_ROOT / "data" / "split_manifest.json").read_text()
    assert committed == manifest_to_json(build_manifest(df))
    m = json.loads(committed)
    assert [m["splits"][k]["n_patients"] for k in ("train", "val", "test")] == [37, 8, 8]
