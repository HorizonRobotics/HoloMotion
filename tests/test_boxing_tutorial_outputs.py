"""Tutorial outputs must not make an otherwise clean image build dirty."""

import subprocess
from pathlib import Path


def test_generated_tutorial_directories_are_ignored_without_hiding_source():
    root = Path(__file__).resolve().parents[1]
    generated = {
        "models/holomotion-v2/config.yaml",
        "results/validation.json",
        "holoboxing/data/raw/InterHuman/download.json",
        "holoboxing/data/converted/InterHuman/motions.json",
        "holoboxing/data/processed/combined/COMPLETE.json",
    }
    source = "holoboxing/data/records.py"
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=root,
        input="\n".join(sorted(generated) + [source]) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    assert set(result.stdout.splitlines()) == generated
