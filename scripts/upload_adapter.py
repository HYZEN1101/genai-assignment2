"""Upload a LoRA adapter folder (+ its config and a short model card) to the Hugging Face Hub.

  python scripts/upload_adapter.py --adapter /content/drive/MyDrive/genai_a2/runs/A/best --run-dir /content/drive/MyDrive/genai_a2/runs/A \
         --repo-id <hf_user>/trialmatch-qwen2.5-1.5b-lora
Needs the HF_TOKEN secret (write access). --repo-id defaults to <whoami>/trialmatch-qwen2.5-1.5b-lora.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
base_model: {base}
library_name: peft
license: apache-2.0
---
# TrialMatch LoRA adapter ({run})

LoRA adapter for `{base}` fine-tuned on the training split (37 patients, seed 42) of ncbi/TrialGPT-Criterion-Annotations:
input = patient note (numbered sentences) + criterion type + criterion text, output = JSON
`{{"label": ..., "evidence_ids": [...]}}`. Educational screening prototype, not a clinical enrollment decision.

Load: `PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{base}", dtype=torch.float16), "<this repo>")` then `merge_and_unload()`.
Training summary: see `summary.json`; config: `config.yaml`.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--run-dir", required=True, help="run folder holding config.yaml and summary.json")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set (Colab secret HF_TOKEN with write access)")
    api = HfApi(token=token)
    repo_id = a.repo_id or f"{api.whoami()['name']}/trialmatch-qwen2.5-1.5b-lora"
    api.create_repo(repo_id, exist_ok=True, private=a.private)
    run = Path(a.run_dir)
    summ = json.loads((run / "summary.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for f in Path(a.adapter).iterdir():
            if f.name in ("state.pt",):          # optimizer state is not part of the release
                continue
            shutil.copy(f, td / f.name)
        shutil.copy(run / "config.yaml", td / "config.yaml")
        shutil.copy(run / "summary.json", td / "summary.json")
        (td / "README.md").write_text(CARD.format(base=summ["model_id"], run=summ["run"]))
        api.upload_folder(folder_path=str(td), repo_id=repo_id, commit_message=f"adapter {summ['run']}")
    print("uploaded: https://huggingface.co/" + repo_id)


if __name__ == "__main__":
    main()
