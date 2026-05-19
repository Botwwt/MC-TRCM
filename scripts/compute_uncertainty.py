from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "final"


@dataclass
class SeedAggregate:
    mean: float
    sd: float
    se: float
    n: int


def ensure_dirs() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def aggregate(values: pd.Series) -> SeedAggregate:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    numeric = numeric[numeric.map(math.isfinite)]
    if numeric.empty:
        return SeedAggregate(math.nan, math.nan, math.nan, 0)
    mean = float(numeric.mean())
    if len(numeric) == 1:
        return SeedAggregate(mean, math.nan, math.nan, 1)
    sd = float(numeric.std(ddof=1))
    return SeedAggregate(mean, sd, float(sd / math.sqrt(len(numeric))), int(len(numeric)))


def main() -> None:
    ensure_dirs()
    per_seed_path = RESULT_ROOT / "per_seed_primary_metrics.csv"
    if not per_seed_path.exists():
        raise FileNotFoundError(
            f"Missing {per_seed_path}. Generate per-seed primary metrics before running uncertainty summaries."
        )
    per_seed = pd.read_csv(per_seed_path)
    rows = []
    for keys, group in per_seed.groupby(["model_name", "dataset_id", "task_name", "metric"], dropna=False):
        agg = aggregate(group["test_value"])
        rows.append(
            {
                "model_name": keys[0],
                "dataset_id": keys[1],
                "task_name": keys[2],
                "metric": keys[3],
                "n_seeds": agg.n,
                "mean": agg.mean,
                "sd": agg.sd,
                "se": agg.se,
                "se_policy": "undefined for n=1" if agg.n <= 1 else "sd/sqrt(n)",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_ROOT / "seed_uncertainty.csv", index=False)
    warnings = []
    if (out["n_seeds"] <= 1).any():
        warnings.append("One or more rows have n_seeds <= 1; seed SE is not estimable.")
    (RESULT_ROOT / "uncertainty_warnings.txt").write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")
    print(out.to_string(index=False))
    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
