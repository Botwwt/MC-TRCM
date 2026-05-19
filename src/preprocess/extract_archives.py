from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.base import list_dataset_ids, load_dataset_config
from src.utils.constants import PROJECT_ROOT
from src.utils.io import ensure_dir, write_json


def parse_dataset_selection(raw_value: str) -> list[str]:
    if raw_value == "all":
        return list_dataset_ids()
    selected = [item.strip() for item in raw_value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(list_dataset_ids()))
    if unknown:
        raise ValueError(f"Unknown dataset ids: {unknown}")
    return selected


def safe_extract_zip(archive_path: Path, destination_dir: Path) -> dict:
    ensure_dir(destination_dir)
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()
        top_level_entries = sorted({Path(info.filename).parts[0] for info in members if info.filename})
        for member in members:
            member_path = destination_dir / member.filename
            resolved_destination = member_path.resolve()
            if destination_dir.resolve() not in resolved_destination.parents and resolved_destination != destination_dir.resolve():
                raise ValueError(f"Unsafe archive member path: {member.filename}")
        zf.extractall(destination_dir)
    return {
        "archive": str(archive_path.relative_to(PROJECT_ROOT)),
        "destination": str(destination_dir.relative_to(PROJECT_ROOT)),
        "member_count": len(members),
        "top_level_entries": top_level_entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract downloaded zip archives into per-dataset extracted directories.")
    parser.add_argument("--dataset", default="all", help="Dataset id or comma-separated ids, or 'all'.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip extraction if destination already contains files.")
    args = parser.parse_args()

    selected_datasets = parse_dataset_selection(args.dataset)
    manifest_records: list[dict] = []

    for dataset_id in selected_datasets:
        config = load_dataset_config(dataset_id)
        raw_dir = ensure_dir(PROJECT_ROOT / config["raw_dir"])
        extracted_dir = ensure_dir(raw_dir / "extracted")
        record = {
            "dataset_id": dataset_id,
            "raw_dir": str(raw_dir.relative_to(PROJECT_ROOT)),
            "extracted_dir": str(extracted_dir.relative_to(PROJECT_ROOT)),
            "archives": [],
        }

        archive_paths = sorted(raw_dir.glob("*.zip"))
        if args.skip_existing and any(extracted_dir.iterdir()):
            record["status"] = "skipped_existing"
            manifest_records.append(record)
            continue

        record["status"] = "no_archives"
        for archive_path in archive_paths:
            record["status"] = "extracted"
            record["archives"].append(safe_extract_zip(archive_path, extracted_dir))
        manifest_records.append(record)

    write_json(
        PROJECT_ROOT / "outputs" / "logs" / "archive_extraction_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "datasets": selected_datasets,
            "records": manifest_records,
        },
    )

    for record in manifest_records:
        archive_count = len(record["archives"])
        print(f"{record['dataset_id']}: status={record['status']} archives={archive_count}")
        for archive in record["archives"]:
            print(
                f"  - {archive['archive']} -> {archive['destination']} "
                f"(members={archive['member_count']}, top_level={archive['top_level_entries'][:5]})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
