
## D16 (P3) Authoritative metric conventions
Own implementation (eval/metrics.py, eval/evidence.py), cross-checked against sklearn (tests) and against the P2 quick metrics on the saved
validation predictions (exact match, tests/test_p3_crosscheck.py). Headline macro-F1 = mean over classes in gold OR valid predictions
(sklearn default, same as the P2 selection rule). Secondary macro_f1_gold_support = classes with gold support > 0 (the test split has no gold
"excluded"). Invalid output = wrong, own confusion column, reported as invalid_rate. Evidence: frozen P1 empty-set policy; HEADLINE = micro P/R/F1
over examples with non-empty gold evidence; "all examples" scope and macro variants also reported.

## D17 (P3) Test-set freeze is enforced in code
scripts/freeze.py writes results/logs/FREEZE.json/.txt (git commit, sha256 of test.jsonl and of the Hub-downloaded adapter weights, decoding
config, versions) and refuses a dirty tree. run_eval.py --split test additionally refuses to run without a matching FREEZE.json
(--allow-test + non-existing output file were already required). Base and fine-tuned models run on identical records with identical decoding.

## D18 (P3) Aggregation rule (pipeline/aggregate.py)
First match wins: inclusion "not included" or exclusion "excluded" -> Likely ineligible; else any "not enough information"/invalid -> Needs more
information; else >=1 decided criterion -> Potentially eligible (preliminary); else No applicable criteria evaluated. Not a ground-truth label.
Known weakness: the model recalls "not included"/"excluded" poorly, so "Potentially eligible" can be wrong.

## D19 (P3) Patient-cluster bootstrap (OPTIONAL, cheap)
1000 resamples of the 8 patients (seed 42), paired for ft vs base, only to show uncertainty (8 patients, rare classes with 0-7 examples).
