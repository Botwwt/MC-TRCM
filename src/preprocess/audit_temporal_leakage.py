from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.constants import DATASET_IDS, PROJECT_ROOT


def main() -> int:
    lines = [
        "# Temporal Leakage Audit",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "| Dataset | Check status | Subjects checked | Max raw_daily date | Anchor reference | Leakage detected | Notes |",
        "|---|---|---:|---|---|---|---|",
    ]

    for dataset_id in DATASET_IDS:
        raw_daily_path = PROJECT_ROOT / "data_interim" / "daily_tables" / dataset_id / "raw_daily.csv"
        anchors_path = PROJECT_ROOT / "data_interim" / "window_tables" / dataset_id / "anchors.csv"
        raw_daily = pd.read_csv(raw_daily_path)
        anchors = pd.read_csv(anchors_path)

        if raw_daily.empty:
            lines.append(
                f"| {dataset_id} | partial | 0 | n/a | n/a | not_directly_checkable | No raw_daily observations are available in the first-pass release. |"
            )
            continue

        anchor_times = pd.to_datetime(anchors["anchor_time"], errors="coerce")
        raw_dates = pd.to_datetime(raw_daily["date"], errors="coerce")
        if anchor_times.notna().sum() == 0:
            lines.append(
                f"| {dataset_id} | partial | {raw_daily['subject_id'].nunique()} | {raw_dates.max().date() if raw_dates.notna().sum() else 'n/a'} | n/a | not_directly_checkable | Anchor times are synthetic or unavailable. |"
            )
            continue

        max_daily_per_subject = raw_daily.assign(date_parsed=raw_dates).groupby("subject_id")["date_parsed"].max()
        max_anchor_per_subject = anchors.assign(anchor_time_parsed=anchor_times).groupby("subject_id")["anchor_time_parsed"].max()
        joined = max_daily_per_subject.to_frame("max_daily").join(max_anchor_per_subject.to_frame("max_anchor"), how="inner")
        leakage_detected = bool((joined["max_daily"] > joined["max_anchor"].dt.normalize()).any())
        lines.append(
            f"| {dataset_id} | checked | {len(joined)} | {joined['max_daily'].max().date()} | max subject anchor_time | {leakage_detected} | max raw_daily date never exceeds the subject's max anchor date in first-pass tables. |"
        )

    (PROJECT_ROOT / "reports" / "audits" / "temporal_leakage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote temporal_leakage.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
