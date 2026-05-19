from __future__ import annotations

import argparse
import hashlib
import shutil
import ssl
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.base import list_dataset_ids, load_dataset_config
from src.utils.constants import PROJECT_ROOT
from src.utils.io import ensure_dir, write_json

USER_AGENT = "MC-TRCM-reproducibility-workspace/0.1"


def parse_dataset_selection(raw_value: str) -> list[str]:
    if raw_value == "all":
        return list_dataset_ids()
    selected = [item.strip() for item in raw_value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(list_dataset_ids()))
    if unknown:
        raise ValueError(f"Unknown dataset ids: {unknown}")
    return selected


def compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, insecure: bool = False) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    context = None
    if insecure:
        context = ssl._create_unverified_context()
    with urllib.request.urlopen(request, context=context) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official dataset artifacts.")
    parser.add_argument("--dataset", default="all", help="Dataset id or comma-separated ids, or 'all'.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned downloads.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files already present.")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification for environments with broken local certificate stores.",
    )
    args = parser.parse_args()

    selected_datasets = parse_dataset_selection(args.dataset)
    run_manifest: list[dict] = []
    had_failure = False

    for dataset_id in selected_datasets:
        config = load_dataset_config(dataset_id)
        raw_dir = ensure_dir(PROJECT_ROOT / config["raw_dir"])

        for target in config["download_targets"]:
            destination = raw_dir / target["filename"]
            record = {
                "dataset_id": dataset_id,
                "target_name": target["name"],
                "url": target["url"],
                "destination": str(destination),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": None,
                "size_bytes": None,
                "md5": None,
                "expected_md5": target.get("expected_md5"),
                "insecure_ssl": args.insecure,
                "error": None,
            }

            if args.dry_run:
                record["status"] = "planned"
                run_manifest.append(record)
                continue

            if args.skip_existing and destination.exists():
                record["status"] = "skipped_existing"
                record["size_bytes"] = destination.stat().st_size
                record["md5"] = compute_md5(destination)
                run_manifest.append(record)
                continue

            try:
                download_file(target["url"], destination, insecure=args.insecure)
                record["status"] = "downloaded"
                record["size_bytes"] = destination.stat().st_size
                record["md5"] = compute_md5(destination)
            except Exception as exc:  # pragma: no cover - network/environment dependent
                had_failure = True
                record["status"] = "failed"
                record["error"] = str(exc)
            run_manifest.append(record)

    write_json(
        PROJECT_ROOT / "outputs" / "logs" / "download_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dry_run": args.dry_run,
            "datasets": selected_datasets,
            "records": run_manifest,
        },
    )

    for record in run_manifest:
        summary = f"{record['dataset_id']} | {record['target_name']} | {record['status']} | {record['destination']}"
        if record["error"]:
            summary += f" | error={record['error']}"
        print(summary)

    return 1 if had_failure and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
