from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.utils.constants import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command while watching a progress log and auto-stop on inactivity."
    )
    parser.add_argument("--progress-log-path", required=True, type=str)
    parser.add_argument("--stall-seconds", type=int, default=600)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _tail_line(path: Path) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-1] if lines else ""


def main() -> None:
    args = parse_args()
    if not args.command:
        raise ValueError("No child command provided.")

    child_command = list(args.command)
    if child_command and child_command[0] == "--":
        child_command = child_command[1:]
    if not child_command:
        raise ValueError("No child command provided after '--'.")

    progress_log_path = Path(args.progress_log_path)
    if not progress_log_path.is_absolute():
        progress_log_path = PROJECT_ROOT / progress_log_path
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    last_reported_line = ""
    last_progress_time = start_time

    print("[monitored] launching:", subprocess.list2cmdline(child_command), flush=True)
    child = subprocess.Popen(child_command, cwd=PROJECT_ROOT)

    try:
        while True:
            return_code = child.poll()

            if progress_log_path.exists():
                last_modified = progress_log_path.stat().st_mtime
                last_progress_time = max(last_progress_time, last_modified)
                last_line = _tail_line(progress_log_path)
                if last_line and last_line != last_reported_line:
                    print(f"[monitored] {last_line}", flush=True)
                    last_reported_line = last_line

            now = time.time()
            if return_code is not None:
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, child_command)
                print("[monitored] child completed successfully", flush=True)
                return

            if now - last_progress_time > int(args.stall_seconds):
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=20)
                raise RuntimeError(
                    f"Progress log stalled for more than {int(args.stall_seconds)} seconds: {progress_log_path}"
                )

            time.sleep(max(int(args.poll_seconds), 1))
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)


if __name__ == "__main__":
    main()
