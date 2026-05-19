from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    scripts = [
        "src/preprocess/preprocess_depresjon.py",
        "src/preprocess/preprocess_deprest_cat.py",
        "src/preprocess/preprocess_obf.py",
        "src/preprocess/preprocess_psyche_d.py",
        "src/preprocess/preprocess_studentlife.py",
    ]
    for script in scripts:
        print(f"Running {script}")
        completed = subprocess.run([sys.executable, str(ROOT / script)], check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
