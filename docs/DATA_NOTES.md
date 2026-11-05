# DATA_NOTES.md (Phase 1; every number comes from the Colab runs of scripts/inspect_data.py, make_split.py, preprocess.py, token_stats.py)

## Dataset
ncbi/TrialGPT-Criterion-Annotations, one HF split "train": 1,015 rows x 14 columns, 53 patients, 103 trials, one note per patient,
no duplicate annotation_id or duplicate (patient, trial, type, text). Raw table cached at data/raw/trialgpt_criterion_annotations.parquet (~176 kB).

## Column roles
- Used: annotation_id, patient_id, note, trial_id, criterion_type, criterion_text, expert_eligibility, expert_sentences.
- BLACKLISTED (GPT-4 output or derived from it): gpt4_explanation, explanation_correctness, gpt4_sentences, gpt4_eligibility.
  Enforced in src/trialmatch/data/schema.py (GPT4_COLUMNS) + tests/test_leakage.py. Preprocessing reads only USED_COLUMNS.
- Ignored: `training` (original TrialGPT flag; we use our own split), trial_title (not part of the required input).

## Labels (expert_eligibility, exact strings; never collapsed)
inclusion: included (150), not included (23), not enough information (174), not applicable (9)  [356 rows]
exclusion: excluded (10), not excluded (484), not enough information (119), not applicable (46) [659 rows]
Heavy imbalance (not excluded = 48% of all rows; excluded = 1%): macro-F1 is the headline metric, accuracy alone is misleading.

## Sentence numbering and evidence
Notes number sentences inline ("0. ... 1. ..."), 0-based; 3-14 sentences per note (median 6), 58-235 words. All 53 notes parse with
sequential-marker splitting; no gold id is out of range. Original ids are kept in every record. Known limitation of the splitter: a sentence
containing the next expected number followed by a period (e.g. "... at 2. ") would be mis-split; checked by the out-of-range test (0 violations).
Gold evidence = expert_sentences, a string "[0, 1]"; parsed to sorted unique ints (0 parse failures).
Evidence ids per row: 0 ids: 610 | 1: 302 | 2: 47 | 3: 24 | 4: 15 | 5: 11 | 6: 1 | 7: 3 | 8: 1 | 9: 1.
Empty gold evidence is common: excluded 0%, included 5%, not included 39%, not enough information 63%, not excluded 76%, not applicable 80%.
Empty gold evidence is NOT proof of anything: it is the annotators' "no supporting sentence".

## Cleaning
One row has a null/empty criterion_text: dropped (it is an exclusion row), logged in results/logs/preprocess_dropped.json.
Manifest row counts include it (1,015); processed counts are 602 / 206 / 206 = 1,014.

## Split (seed 42, patient-grouped)
Method: ids = sorted(set(patient_id)) (plain string sort, e.g. 'sigir-20141' < 'sigir-201410'); arr = np.array(ids); rng = np.random.default_rng(42);
rng.shuffle(arr) (in place); train = arr[:37], val = arr[37:45], test = arr[45:53]. Manifest: data/split_manifest.json (no timestamps, byte-identical regeneration,
tested). Rows: train 603 (37 patients), val 206 (8), test 206 (8). patient_assignment_sha256 = 5a5e4c11...2fb9, rows_fingerprint_sha256 = e838374f...ee21.
Processed label counts (train / val / test):
| type | label | train | val | test |
|---|---|---|---|---|
| inclusion | included | 88 | 35 | 27 |
| inclusion | not included | 13 | 3 | 7 |
| inclusion | not enough information | 109 | 27 | 38 |
| inclusion | not applicable | 5 | 1 | 3 |
| exclusion | excluded | 8 | 2 | **0** |
| exclusion | not excluded | 291 | 101 | 91 |
| exclusion | not enough information | 65 | 26 | 28 |
| exclusion | not applicable | 23 | 11 | 12 |
Consequence: the test set has NO "excluded" example and only 3 inclusion "not applicable"; Phase 3 must report per-class support and say which
classes are absent. Rows with empty gold evidence: train 348/602, val 116/206, test 145/206.

## Output schema
{"label": "<dataset label valid for the criterion type>", "evidence_ids": [sorted unique ints]}, e.g. {"label": "not enough information", "evidence_ids": [0]}.
Parser: parse_output(text, criterion_type, n_sentences) -> {label, evidence_ids, valid, error, unknown_ids}. Invalid = no JSON object, label missing / not allowed
for the type, evidence_ids not a list of ints. Invalid outputs count as wrong predictions (rate reported). Out-of-range cited ids are returned in unknown_ids
(hallucinated ids; scored as false-positive evidence), they do not make the output invalid.
Explanations are NOT generated: they are built from the cited sentences by a template (Phase 3); no GPT-4 text anywhere.

## Prompt
Qwen chat template, explicit system message, user message = numbered note "[id] text", criterion type, criterion text, label meanings, allowed labels,
JSON-only instruction (src/trialmatch/data/prompts.py). Identical for base and fine-tuned models. Prompt wording may be revised ONLY using validation data.

## Empty-evidence policy for evidence metrics (frozen in Phase 1, before any test result exists)
Per example with gold set G and predicted set P (sentence ids):
- G and P both empty: P = R = F1 = 1.   - G empty, P non-empty: precision 0, recall undefined -> counted as 1 (nothing to miss), F1 0.
- G non-empty, P empty: recall 0, precision defined as 0, F1 0.
Reported: micro (pooled TP/FP/FN) and macro (mean over examples), each (a) over all examples and (b) over examples with non-empty gold only.
HEADLINE = (b), because (a) is inflated by easy empty/empty cases (70% of test rows have empty gold). Unknown/hallucinated ids count as FP.

## Truncation / token lengths (tokenizer: Qwen2.5, prompt includes chat template)
Untruncated prompt tokens, train / val / test: p50 322 / 364 / 321.5, p95 520 / 412 / 396, p99 566 / 437 / 427, max 629 / 468 / 533. Target max 38 tokens (mean ~15).
max_len = 1024 (reserve 96): 0 records truncated in any split. Had we chosen 512, 73 / 8 / 5 records (train/val/test) would exceed the budget and, under naive tail
cutting, 15 / 1 / 1 would lose gold evidence sentences. Policy implemented anyway (never silent): drop non-protected sentences from the END first, protect gold evidence
(training), insert "(Note: sentences [...] were omitted ...)" into the prompt, record truncated_ids. Figure: results/figures/token_lengths.png.

## Leakage checks
tests/test_leakage.py: forbidden keys, poisoning test (changing gpt4_sentences does not change any record), GPT-4 columns not even read, real-data scan.
Real-data scan finding: exactly one GPT-4 explanation string (annotation 604) equals the boilerplate last sentence of the notes ("The patient will provide informed consent, ..."),
so it appears in 1,014 prompts as legitimate note text, never in a target or criterion. The test therefore allows a GPT-4 string in a prompt only if it is verbatim part of that
record's own note/criterion, and forbids it in targets.
