# Eligibility screening report

> **Educational screening prototype, not a clinical enrollment decision.**

## Overall preliminary outcome: Potentially eligible (preliminary)

The overall outcome is a rule-based aggregation of criterion-level model outputs, not a ground-truth label. 'Potentially eligible (preliminary)' can be wrong because the model misses some 'not included'/'excluded' cases (see error analysis). Absence of evidence is never treated as eligibility or ineligibility.

Rule: Rule (first match wins): (1) any inclusion criterion 'not included' or exclusion criterion 'excluded' -> Likely ineligible; (2) otherwise any criterion 'not enough information' or unparseable -> Needs more information; (3) otherwise, if at least one criterion is decided -> Potentially eligible (preliminary); (4) otherwise -> No applicable criteria evaluated.


## Criterion-level results

### 1. [inclusion] Adults with joint pain

- **Predicted label:** included
- **Cited sentence IDs:** [0]
- **Evidence (verbatim from note):** [0] A 52-year-old woman presents with swelling and pain in both wrists and knees for six months.
- **Explanation (templated):** The cited sentence(s) [0] indicate that the patient meets this inclusion criterion.

### 2. [exclusion] History of chronic systemic autoimmune disorders such as rheumatoid arthritis

- **Predicted label:** not excluded
- **Cited sentence IDs:** [1]
- **Evidence (verbatim from note):** [1] She has long-standing rheumatoid arthritis and takes methotrexate and prednisone 10 mg daily.
- **Explanation (templated):** The cited sentence(s) [1] indicate that the patient does not meet this exclusion criterion.

### 3. [exclusion] Concurrent use of systemic immunosuppressive therapy or long-term corticosteroids

- **Predicted label:** not excluded
- **Cited sentence IDs:** [1]
- **Evidence (verbatim from note):** [1] She has long-standing rheumatoid arthritis and takes methotrexate and prednisone 10 mg daily.
- **Explanation (templated):** The cited sentence(s) [1] indicate that the patient does not meet this exclusion criterion.

### 4. [exclusion] History of malignant neoplasm

- **Predicted label:** not excluded
- **Cited sentence IDs:** [2]
- **Evidence (verbatim from note):** [2] She has no history of cancer.
- **Explanation (templated):** The cited sentence(s) [2] indicate that the patient does not meet this exclusion criterion.

## Patient note (as numbered for the model)

[0] A 52-year-old woman presents with swelling and pain in both wrists and knees for six months.
[1] She has long-standing rheumatoid arthritis and takes methotrexate and prednisone 10 mg daily.
[2] She has no history of cancer.
[3] Vital signs are stable.
[4] The patient is willing to participate in the trial.

## Provenance

- generated_utc: 2026-11-10 17:50:44 UTC
- model_id: Qwen/Qwen2.5-1.5B-Instruct
- adapter: C:\Users\Hp\Desktop\genai_a2\genai-assignment2\weights\adapter
- adapter_sha256: 6edf003f9307ee69b5af5a007db8944b153859418032a2361dbf08c6475dd051
- device: cpu
- dtype: float32
- decoding_mode: KV cache
- speculative_k: n/a
- decoding: greedy, max_new_tokens=64
- input_sha256: 72fe8938dd74ebc056b34b0c85b7f8413b9a3bbdd937bfb2e7171c3fcf8611dd
- note_numbering: already numbered (dataset format '0. ... 1. ...')
- prompt_token_budget: 1024
- note_truncated_for_some_criteria: False
- versions: {'torch': '2.15.0+cpu', 'transformers': '5.22.2', 'peft': '0.21.2'}

_Educational screening prototype, not a clinical enrollment decision._
