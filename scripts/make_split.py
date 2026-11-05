"""Phase 1 step 2: create / verify data/split_manifest.json. Run: python scripts/make_split.py"""
from __future__ import annotations

import sys
from pathlib import Path

from trialmatch.data.load import REPO_ROOT, load_raw
from trialmatch.data.split import build_manifest, manifest_to_json, verify_manifest

path = REPO_ROOT / "data" / "split_manifest.json"
df = load_raw()
m = build_manifest(df)
verify_manifest(m, df)
text = manifest_to_json(m)
if path.exists() and path.read_text() != text:
    sys.exit(f"ERROR: {path} exists and differs from regeneration. Investigate; do NOT overwrite silently.")
path.write_text(text)
print(f"wrote {path} ({len(text)} bytes)")
for k, v in m["splits"].items():
    print(f"{k:5s}: {v['n_patients']:2d} patients, {v['n_rows']:4d} rows | first ids: {v['patient_ids'][:3]}")
print("patient_assignment_sha256:", m["patient_assignment_sha256"])
print("rows_fingerprint_sha256  :", m["rows_fingerprint_sha256"])
