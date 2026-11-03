# scaffold.py: creates the repo layout. Safe to re-run (never overwrites existing files).
from pathlib import Path

DIRS = [
    "configs", "data/processed",
    "src/trialmatch/data", "src/trialmatch/train", "src/trialmatch/eval",
    "src/trialmatch/attention", "src/trialmatch/decoding", "src/trialmatch/pipeline",
    "app/assets", "app/sample_trials", "scripts", "notebooks", "tests",
    "results/logs", "results/tables", "results/figures", "results/predictions", "results/benchmarks",
    "docs/handoffs", "report/figures",
]
PKGS = ["", "data", "train", "eval", "attention", "decoding", "pipeline"]
TESTS = ["test_split", "test_leakage", "test_metrics", "test_attention", "test_kv_equiv", "test_spec_equiv"]

REQUIREMENTS = """torch
transformers
peft
accelerate
datasets
numpy
pandas
scikit-learn
pyyaml
pytest
gradio
matplotlib
huggingface_hub
psutil
"""

GITIGNORE = """.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
# large artifacts live on Hugging Face Hub / Drive, never in git
*.safetensors
*.bin
*.pt
*.gguf
.env
.DS_Store
report/*.aux
report/*.log
report/*.out
"""

PYPROJECT = """[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "trialmatch"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

README = """# TrialMatch (GenAI Assignment 2)

Educational screening prototype, **not a clinical enrollment decision**.

Status: scaffold only. Setup, data, training, evaluation, benchmarks and app instructions are added per phase.
"""

LITERATURE = """# Literature table

| Citation | URL | Contribution | Limitation | Use here | Verified by me |
|---|---|---|---|---|---|
"""

FILES = {
    "requirements.txt": REQUIREMENTS,
    ".gitignore": GITIGNORE,
    "pyproject.toml": PYPROJECT,
    "README.md": README,
    "docs/DECISIONS.md": "# Decisions log (chosen, rejected, why, evidence)\n\n",
    "docs/REPORT_NOTES.md": "# Report notes (append one block per phase)\n\n",
    "docs/literature.md": LITERATURE,
    "docs/model_selection.md": "# Model selection\n\nTBD (Phase 0)\n",
    "tests/test_smoke.py": "def test_smoke():\n    assert True\n",
}


def main() -> None:
    for d in DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
        if d in ("configs", "data/processed", "app/assets", "app/sample_trials", "docs/handoffs",
                 "report/figures") or d.startswith("results/"):
            (Path(d) / ".gitkeep").touch(exist_ok=True)
    for p in PKGS:
        (Path("src/trialmatch") / p / "__init__.py").touch(exist_ok=True)
    for t in TESTS:
        f = Path("tests") / f"{t}.py"
        if not f.exists():
            f.write_text("# placeholder, filled in its phase\n")
    for name, content in FILES.items():
        f = Path(name)
        if not f.exists():
            f.write_text(content)
    print("scaffold done")


if __name__ == "__main__":
    main()
