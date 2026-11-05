"""Shared synthetic fixtures (no network, no tokenizer download)."""
import pandas as pd
import pytest


class FakeTok:
    """Whitespace 'tokenizer' with a chat-template-like renderer; enough to test prompt/truncation logic."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        class O: pass
        o = O(); o.input_ids = text.split(); return o


def make_note(n):
    return " ".join(f"{i}. Sentence number {i} about the patient, aged 58. Hb low." for i in range(n))


@pytest.fixture
def synth_df():
    rows, aid = [], 0
    for p in range(53):
        note = make_note(6 + p % 4)
        for t in range(3):
            for ctype, label, ev in (("inclusion", "included", "[0, 1]"), ("exclusion", "not excluded", "[]"),
                                     ("inclusion", "not enough information", "[2]")):
                rows.append(dict(annotation_id=aid, patient_id=f"sigir-2014{p}", note=note, trial_id=f"NCT{t}",
                                 trial_title="T", criterion_type=ctype, criterion_text=f"criterion {aid}",
                                 gpt4_explanation=f"SECRET GPT4 EXPLANATION number {aid} xyz", explanation_correctness="Correct",
                                 gpt4_sentences="[0]", expert_sentences=ev, gpt4_eligibility="included",
                                 expert_eligibility=label, training=True))
                aid += 1
    return pd.DataFrame(rows)
