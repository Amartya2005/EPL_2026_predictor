"""
Run the full offline pipeline: load raw data -> standings -> features ->
clustering -> Elo -> model training. Produces every CSV/JSON/pickle the
Streamlit app reads from data/processed and models/.

Usage:
    python scripts/build_all.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ROOT / "src" / "data" / "load_data.py",
    ROOT / "src" / "data" / "features.py",
    ROOT / "src" / "models" / "clustering.py",
    ROOT / "src" / "models" / "elo.py",
    ROOT / "src" / "models" / "train.py",
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n{'=' * 70}\nRunning {step.relative_to(ROOT)}\n{'=' * 70}")
        result = subprocess.run([sys.executable, str(step)], cwd=str(step.parent))
        if result.returncode != 0:
            print(f"FAILED at {step}")
            sys.exit(1)
    print("\nAll pipeline steps completed. Artifacts in data/processed/ and models/.")
