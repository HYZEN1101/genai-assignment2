# Eligibility screening report

> **Educational screening prototype, not a clinical enrollment decision.**

## Overall preliminary outcome: Needs more information

The overall outcome is a rule-based aggregation of criterion-level model outputs, not a ground-truth label. 'Potentially eligible (preliminary)' can be wrong because the model misses some 'not included'/'excluded' cases (see error analysis). Absence of evidence is never treated as eligibility or ineligibility.

Rule: Rule (first match wins): (1) any inclusion criterion 'not included' or exclusion criterion 'excluded' -> Likely ineligible; (2) otherwise any criterion 'not enough information' or unparseable -> Needs more information; (3) otherwise, if at least one criterion is decided -> Potentially eligible (preliminary); (4) otherwise -> No applicable criteria evaluated.

Unresolved criteria (missing information / not evaluated): #2

## Criterion-level results

### 1. [inclusion] Age 40 years or older

- **Predicted label:** included
- **Cited sentence IDs:** [0]
- **Evidence (verbatim from note):** [0] A 67-year-old man presents with a three-day history of productive cough and shortness of breath.
- **Explanation (templated):** The cited sentence(s) [0] indicate that the patient meets this inclusion criterion.

### 2. [inclusion] Diagnosis of chronic obstructive pulmonary disease confirmed by spirometry

- **Predicted label:** not enough information  (missing information)
- **Cited sentence IDs:** [0, 1]
- **Evidence (verbatim from note):** [0] A 67-year-old man presents with a three-day history of productive cough and shortness of breath. | [1] He has smoked one pack of cigarettes daily for 40 years.
- **Explanation (templated):** The note does not contain enough information to decide this criterion; missing information is not treated as eligibility or ineligibility. Related, non-conclusive sentence(s): [0, 1].

### 3. [inclusion] Current or former smoker with at least 10 pack-years

- **Predicted label:** included
- **Cited sentence IDs:** [1]
- **Evidence (verbatim from note):** [1] He has smoked one pack of cigarettes daily for 40 years.
- **Explanation (templated):** The cited sentence(s) [1] indicate that the patient meets this inclusion criterion.

### 4. [exclusion] Currently pregnant or breastfeeding

- **Predicted label:** not applicable
- **Cited sentence IDs:** []
- **Evidence (verbatim from note):** (none cited)
- **Explanation (templated):** The model judged this criterion not applicable to this patient.

### 5. [exclusion] Participation in another interventional trial within 30 days

- **Predicted label:** not excluded
- **Cited sentence IDs:** []
- **Evidence (verbatim from note):** (none cited)
- **Explanation (templated):** The model labelled this criterion 'not excluded' but cited no supporting sentence. The absence of a mention is not proof either way; verify against the full record.

### 6. [exclusion] Known allergy to inhaled corticosteroids

- **Predicted label:** not excluded
- **Cited sentence IDs:** []
- **Evidence (verbatim from note):** (none cited)
- **Explanation (templated):** The model labelled this criterion 'not excluded' but cited no supporting sentence. The absence of a mention is not proof either way; verify against the full record.

## Patient note (as numbered for the model)

[0] A 67-year-old man presents with a three-day history of productive cough and shortness of breath.
[1] He has smoked one pack of cigarettes daily for 40 years.
[2] On examination he has diffuse wheezing and oxygen saturation of 91% on room air.

## Provenance

- generated_utc: 2026-11-10 18:13:41 UTC
- model_id: Qwen/Qwen2.5-1.5B-Instruct
- adapter: C:\Users\Hp\Desktop\genai_a2\genai-assignment2\weights\adapter
- adapter_sha256: 6edf003f9307ee69b5af5a007db8944b153859418032a2361dbf08c6475dd051
- device: cpu
- dtype: float32
- decoding_mode: KV cache
- speculative_k: n/a
- decoding: greedy, max_new_tokens=64
- input_sha256: d49cf16748468f5d25460b925fe6eaabac6ca25be1759e8a1f34e799d078e3a9
- note_numbering: already numbered (dataset format '0. ... 1. ...')
- prompt_token_budget: 1024
- note_truncated_for_some_criteria: False
- versions: {'torch': '2.15.0+cpu', 'transformers': '5.22.2', 'peft': '0.21.2'}

_Educational screening prototype, not a clinical enrollment decision._
