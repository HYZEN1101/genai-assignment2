# results/app_tests.md : unseen-case test sheet (fill in from the REAL app run; paste actual outputs, do not edit them)
Setup: date ___ | machine ___ (CPU/GPU, RAM) | mode ___ | k ___ | max new tokens ___ | model load time ___ s

| # | Case (what it tests) | Note (short) | Criteria | Result per criterion (label, cited IDs) | Overall outcome | Judgement (correct / wrong / why) | Seconds |
|---|---|---|---|---|---|---|---|
| 1 | short note, many criteria | "A 30-year-old woman reports two weeks of fever. She is not pregnant." | incl: Age 18 or older; incl: Fever documented; excl: Pregnancy; excl: Known HIV infection | | | | |
| 2 | missing information | "A 70-year-old man had a hip fracture repaired last week." | incl: Age >= 65; incl: eGFR >= 60; excl: Anticoagulant therapy | | | | |
| 3 | contradictory statements | "0. The patient has never smoked. 1. She reports smoking ten cigarettes a day." | incl: Current smoker | | | | |
| 4 | not applicable | "A 55-year-old man with type 2 diabetes. HbA1c 8.1%." | incl: Women of childbearing potential use contraception; incl: HbA1c >= 7% | | | | |
| 5 | long note (>1024 tokens -> truncation warning expected) | paste ~40 sentences | incl: any | | | | |
| 6 | different style (telegraphic) | "58M, HTN, DM2, eGFR 38, on lisinopril. No chest pain." | excl: Severe renal impairment (eGFR < 30); incl: Hypertension | | | | |
| 7 | pasted trial text with headers | "Inclusion Criteria: ... Exclusion Criteria: ..." | use the split button | | | | |

Bad-input checks (expected: clear message, session continues): empty note | note > 12,000 chars | no criteria | > 20 criteria | non-English note | criterion number out of range in tab 2 | Run clicked twice quickly (busy message).
Live comparison check (tab 2): modes ___ | tokens equal? ___ | prefill/decode ms ___ | speculative acceptance ___ | speedup vs KV ___ (a slowdown on CPU is expected; see P6).
Report export check: HTML opens offline = ___ | Markdown = ___ | provenance block present = ___
