"""Download the released LoRA adapter into weights/adapter (works logged-out if the repo is public).

  python scripts/download_weights.py [--repo-id user/name] [--out weights/adapter]
Default repo id: adapter_repo in configs/eval.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--out", default=str(ROOT / "weights" / "adapter"))
    a = ap.parse_args()
    repo = a.repo_id or yaml.safe_load(open(ROOT / "configs" / "eval.yaml"))["adapter_repo"]
    if not repo or repo == "TBD":
        raise SystemExit("no adapter repo id: set adapter_repo in configs/eval.yaml or pass --repo-id")
    path = snapshot_download(repo_id=repo, local_dir=a.out)
    print("adapter downloaded to", path)


if __name__ == "__main__":
    main()
