import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path | str, payload: dict | list) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path | str, frame: pd.DataFrame) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def write_parquet_with_fallback(path: Path | str, frame: pd.DataFrame) -> dict:
    path = Path(path)
    ensure_dir(path.parent)
    record = {
        "requested_path": str(path),
        "written_path": None,
        "format": None,
        "fallback_reason": None,
    }
    try:
        frame.to_parquet(path, index=False)
        record["written_path"] = str(path)
        record["format"] = "parquet"
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        fallback_path = path.with_suffix(".csv")
        frame.to_csv(fallback_path, index=False)
        record["written_path"] = str(fallback_path)
        record["format"] = "csv_fallback"
        record["fallback_reason"] = str(exc)
    return record
