from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_TABLES = ROOT / "tables" / "core_protocol"
LEGACY_TABLES = ROOT / "tables" / "final"
MAIN_TABLES = [
    CORE_TABLES / "core_main_comparison_clean.tex",
    CORE_TABLES / "core_calibration_comparison.tex",
    CORE_TABLES / "recursive_depth_selection.tex",
    CORE_TABLES / "ablation_compact_summary.tex",
    CORE_TABLES / "feature_source_core_summary.tex",
    CORE_TABLES / "optimization_ablation_summary.tex",
    CORE_TABLES / "appendix_baseline_full_regression_metrics.tex",
    CORE_TABLES / "appendix_baseline_full_classification_metrics.tex",
    CORE_TABLES / "appendix_feature_source_baseline_primary_metrics.tex",
]

BANNED_MANUSCRIPT_PATTERNS = [
    (re.compile(r"context-variable dominance", re.IGNORECASE), "context-variable dominance framing"),
    (re.compile(r"\blocked benchmark\b", re.IGNORECASE), "locked benchmark language"),
    (re.compile(r"\blocked exports\b", re.IGNORECASE), "locked exports language"),
    (re.compile(r"revised diagnostic suite", re.IGNORECASE), "diagnostic-suite run-management language"),
    (re.compile(r"debugging evidence only", re.IGNORECASE), "debugging-evidence language"),
    (re.compile(r"\bnot promoted\b", re.IGNORECASE), "not-promoted language"),
    (re.compile(r"does not support .*superiority", re.IGNORECASE), "self-defeating superiority framing"),
    (re.compile(r"\bsame-test\b", re.IGNORECASE), "same-test language"),
    (re.compile(r"results/revised", re.IGNORECASE), "local result-path exposure"),
    (re.compile(r"\bK=1 model\b", re.IGNORECASE), "K=1 model-family language"),
    (re.compile(r"\bK=4 locked\b", re.IGNORECASE), "locked K language"),
]


def table_cells(line: str) -> list[str]:
    if "&" not in line or line.lstrip().startswith("%"):
        return []
    return [cell.strip().rstrip("\\").strip() for cell in line.split("&")]


def has_number(text: str) -> bool:
    return bool(re.search(r"[-+]?\d+\.\d+", text))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate manuscript table/style invariants.")
    parser.add_argument("--skip-log", action="store_true", help="Skip checks that require a fresh LaTeX compile log.")
    parser.add_argument(
        "--legacy-final-tables",
        action="store_true",
        help="Also inspect older tables/final assets that are not used by the rewritten manuscript.",
    )
    args = parser.parse_args()
    failures: list[str] = []
    table_root = LEGACY_TABLES if args.legacy_final_tables else CORE_TABLES
    tex_files = [ROOT / "main.tex", *table_root.glob("*.tex")]

    for path in tex_files:
        text = path.read_text(encoding="utf-8")
        if "TODO" in text:
            failures.append(f"{path.relative_to(ROOT)} contains TODO text")
        if "\u00b1" in text or "\u5364" in text:
            failures.append(f"{path.relative_to(ROOT)} contains non-LaTeX plus/minus formatting")
        if path.name == "main.tex":
            bibliography_index = text.find(r"\bibliography{references}")
            appendix_index = text.find(r"\appendix")
            if bibliography_index == -1:
                failures.append("main.tex is missing the references bibliography command")
            if appendix_index == -1:
                failures.append("main.tex is missing the appendix marker")
            if bibliography_index != -1 and appendix_index != -1 and bibliography_index > appendix_index:
                failures.append("main.tex places the appendix before references")
            for pattern, description in BANNED_MANUSCRIPT_PATTERNS:
                for match in pattern.finditer(text):
                    line_no = text[: match.start()].count("\n") + 1
                    failures.append(f"{path.relative_to(ROOT)}:{line_no} contains {description}")

        for line_no, line in enumerate(text.splitlines(), start=1):
            for cell in table_cells(line):
                if "K=1 model" in cell and has_number(cell):
                    failures.append(f"{path.relative_to(ROOT)}:{line_no} combines K=1 model label with numeric cell: {cell}")
                if "MC-TRCM K=1 model" in cell and has_number(cell):
                    failures.append(f"{path.relative_to(ROOT)}:{line_no} combines MC-TRCM K=1 model label with numeric cell: {cell}")
                if r"\ModelName{} $K=1$" in cell and has_number(cell):
                    failures.append(f"{path.relative_to(ROOT)}:{line_no} combines MC-TRCM K=1 macro label with numeric cell: {cell}")
                if re.search(r"\b(elastic_net|simple_multitask|mctrcm_core|locked_mctrcm|k1_mctrcm)\b", cell):
                    failures.append(f"{path.relative_to(ROOT)}:{line_no} contains raw model identifier: {cell}")

    for path in MAIN_TABLES:
        if not path.exists():
            failures.append(f"Missing main table {path.relative_to(ROOT)}")
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip().endswith(r"\\"):
                cells = table_cells(line)
                if any(cell == "" for cell in cells):
                    failures.append(f"{path.relative_to(ROOT)}:{line_no} has an empty table cell")

    required_docs = [
        "STYLE_CHANGELOG.md",
        "TABLE_INVENTORY.md",
        "CLAIMS_CHECKLIST.md",
        "EXPERIMENT_COMPLETION_REPORT.md",
        "experiments_to_run.md",
        "manuscript_changes.md",
    ]
    for name in required_docs:
        if not (ROOT / name).exists():
            failures.append(f"Missing {name}")

    log_path = ROOT / "main.log"
    if log_path.exists() and not args.skip_log:
        log = log_path.read_text(encoding="utf-8", errors="ignore")
        if "LaTeX Warning: There were undefined references" in log:
            failures.append("main.log reports undefined references")
        if "LaTeX Warning: Citation" in log and "undefined" in log:
            failures.append("main.log reports undefined citations")
        if "Overfull \\hbox" in log:
            failures.append("main.log contains overfull hbox warnings; inspect table/page width")
        if "Underfull \\hbox" in log or "Underfull \\vbox" in log:
            failures.append("main.log contains underfull box warnings; inspect spacing")
        if "Infinite glue shrinkage" in log:
            failures.append("main.log contains longtable/page-splitting glue warnings")
        if "pdfTeX warning" in log:
            failures.append("main.log contains pdfTeX warnings")

    if failures:
        print("VALIDATION FAILED")
        for item in failures:
            print(f"- {item}")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
