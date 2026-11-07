"""Authoritative Phase 3 metrics must reproduce the Phase 2 quick metrics on the saved VALIDATION predictions."""
import json
from pathlib import Path

import pytest

from trialmatch.eval.metrics import evaluate_rows

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("pred,quick", [("val_C_ep5", "quick_val_C_ep5"), ("val_base", "quick_val_val_base")])
def test_reproduces_p2_numbers(pred, quick):
    pp, qp = ROOT / "results" / "predictions" / f"{pred}.jsonl", ROOT / "results" / "tables" / f"{quick}.json"
    if not (pp.exists() and qp.exists()):
        pytest.skip("P2 artefacts not present")
    rows = [json.loads(l) for l in open(pp) if l.strip()]
    q = json.loads(qp.read_text())["metrics"]
    e = evaluate_rows(rows)
    assert e["overall"]["accuracy"] == pytest.approx(q["accuracy"], abs=1e-12)
    assert e["overall"]["macro_f1"] == pytest.approx(q["macro_f1"], abs=1e-12)
    assert e["inclusion"]["macro_f1"] == pytest.approx(q["inclusion_macro_f1"], abs=1e-12)
    assert e["exclusion"]["macro_f1"] == pytest.approx(q["exclusion_macro_f1"], abs=1e-12)
    assert e["evidence"]["overall"]["nonempty_gold"]["micro_f1"] == pytest.approx(q["evidence_nonempty_gold"]["micro_f1"], abs=1e-12)
    assert e["evidence"]["overall"]["all"]["macro_f1"] == pytest.approx(q["evidence_all"]["macro_f1"], abs=1e-12)
