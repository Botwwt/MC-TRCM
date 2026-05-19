from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT
PREDICTION_DIR = BUNDLE_ROOT / "outputs" / "predictions" / "mctrcm_v2"
RESULT_ROOT = ROOT / "results" / "mctrcm_core_protocol_v1"
LOG_DIR = BUNDLE_ROOT / "outputs" / "logs"

CORE_TASKS = {
    "deprest_cat": ("phq9_reg", "gad7_reg", "phq9_cat", "gad7_cat"),
    "psyche_d": ("phq_change_binary", "phq_change_multiclass"),
}
CORE_TASK_KEYS = {
    dataset: tuple(f"{dataset}::{task}" for task in tasks)
    for dataset, tasks in CORE_TASKS.items()
}
TASK_DISPLAY = {
    "deprest_cat::phq9_reg": "DepreST-CAT PHQ-9 severity",
    "deprest_cat::gad7_reg": "DepreST-CAT GAD-7 severity",
    "deprest_cat::phq9_cat": "DepreST-CAT PHQ-9 category",
    "deprest_cat::gad7_cat": "DepreST-CAT GAD-7 category",
    "psyche_d::phq_change_binary": "PSYCHE-D PHQ-change binary",
    "psyche_d::phq_change_multiclass": "PSYCHE-D PHQ-change multiclass",
}
REGRESSION_TASKS = {"deprest_cat::phq9_reg", "deprest_cat::gad7_reg"}
CLASSIFICATION_TASKS = tuple(
    task_key
    for task_keys in CORE_TASK_KEYS.values()
    for task_key in task_keys
    if task_key not in REGRESSION_TASKS
)
TABULAR_BASELINE_FAMILIES = {"elastic_net", "lightgbm", "xgboost", "ebm"}
NEURAL_BASELINE_FAMILIES = {"mlp", "simple_multitask_mlp", "gru", "lstm", "transformer"}
NULL_BASELINE_FAMILIES = {"null"}

SEARCH_SPACE = {
    "learning_rate": (1e-4, 2e-4, 4.5e-4, 8e-4, 1e-3),
    "weight_decay": (1e-5, 5e-5, 1.5e-4, 5e-4),
    "dropout": (0.0, 0.1, 0.2, 0.35),
    "token_dim": (64, 96, 128),
    "latent_dim": (96, 128, 192),
    "output_refine_dim": (16, 32, 64),
    "batch_size": (64, 128, 256),
    "patience": (8, 12, 16),
    "regression_loss": ("huber", "mse", "mae_huber"),
    "film_conditioning_mode": ("both", "pre", "post"),
    "conditioning_mode": ("dataset_task",),
    "modality_dropout": (0.0, 0.1, 0.2),
}

COMPACT_CANDIDATES = (
    {
        "learning_rate": 4.5e-4,
        "weight_decay": 1.5e-4,
        "dropout": 0.1,
        "token_dim": 64,
        "latent_dim": 96,
        "output_refine_dim": 16,
        "batch_size": 128,
        "patience": 8,
        "regression_loss": "huber",
        "film_conditioning_mode": "both",
        "conditioning_mode": "dataset_task",
        "modality_dropout": 0.0,
    },
    {
        "learning_rate": 2e-4,
        "weight_decay": 5e-5,
        "dropout": 0.2,
        "token_dim": 96,
        "latent_dim": 128,
        "output_refine_dim": 32,
        "batch_size": 128,
        "patience": 12,
        "regression_loss": "huber",
        "film_conditioning_mode": "both",
        "conditioning_mode": "dataset_task",
        "modality_dropout": 0.1,
        "missingness_embedding_norm": True,
    },
    {
        "learning_rate": 8e-4,
        "weight_decay": 5e-4,
        "dropout": 0.35,
        "token_dim": 64,
        "latent_dim": 128,
        "output_refine_dim": 32,
        "batch_size": 256,
        "patience": 12,
        "regression_loss": "mae_huber",
        "film_conditioning_mode": "pre",
        "conditioning_mode": "dataset_task",
        "modality_dropout": 0.1,
        "class_balance_mode": "effective_number",
    },
    {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.0,
        "token_dim": 128,
        "latent_dim": 192,
        "output_refine_dim": 64,
        "batch_size": 64,
        "patience": 16,
        "regression_loss": "mse",
        "film_conditioning_mode": "post",
        "conditioning_mode": "dataset_task",
        "modality_dropout": 0.0,
    },
    {
        "learning_rate": 1e-3,
        "weight_decay": 1.5e-4,
        "dropout": 0.2,
        "token_dim": 96,
        "latent_dim": 96,
        "output_refine_dim": 16,
        "batch_size": 256,
        "patience": 8,
        "regression_loss": "huber",
        "film_conditioning_mode": "both",
        "conditioning_mode": "dataset_task",
        "modality_dropout": 0.2,
        "binary_focal_gamma": 1.5,
        "multiclass_label_smoothing": 0.05,
    },
)

ABLATION_FLAGS = {
    "full": (),
    "no_film": ("--film-conditioning-mode", "none"),
    "no_missingness_token": ("--disable-missingness-tokens",),
    "no_missingness_projection": ("--disable-missingness-embedding",),
    "no_pcgrad": ("--disable-pcgrad",),
    "gradnorm": ("--use-gradnorm",),
    "no_uncertainty_weighting": ("--disable-uncertainty-weighting",),
    "no_task_balanced_sampling": ("--task-sampling-power", "0.0", "--classification-sampling-power", "0.0"),
    "no_conflict_weighting": ("--disable-pcgrad", "--disable-uncertainty-weighting"),
    "no_task_conditioning": ("--conditioning-mode", "none"),
    "no_recursive_refinement": (),
}

FEATURE_SOURCE_FLAGS = {
    "full": ("--feature-source-condition", "FULL"),
    "sensor_values_only": (
        "--feature-source-condition",
        "SENSOR_VALUES_ONLY",
        "--disable-native-branch",
        "--disable-missingness-tokens",
        "--disable-missingness-embedding",
    ),
    "sensor_plus_missingness": (
        "--feature-source-condition",
        "SENSOR_PLUS_MISSINGNESS",
        "--disable-native-branch",
    ),
    "missingness_only": (
        "--feature-source-condition",
        "MISSINGNESS_ONLY",
        "--disable-native-branch",
    ),
    "symptom_static_clinical": (
        "--feature-source-condition",
        "SYMPTOM_STATIC_CLINICAL",
        "--disable-native-branch",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation-selected MC-TRCM core benchmark protocol."
    )
    parser.add_argument(
        "--stage",
        choices=["smoke", "k_search", "final", "ablations", "feature_sources", "ensemble", "summarize", "all"],
        default="summarize",
    )
    parser.add_argument("--datasets", nargs="*", choices=sorted(CORE_TASKS), default=list(CORE_TASKS))
    parser.add_argument("--k-values", nargs="*", type=int, default=[1, 2, 4, 6, 8])
    parser.add_argument("--search-seeds", nargs="*", type=int, default=[20260417])
    parser.add_argument("--final-seeds", nargs="*", type=int, default=[20260417, 20260418, 20260419, 20260420, 20260421])
    parser.add_argument("--search-profile", choices=["compact", "random", "full_grid"], default="compact")
    parser.add_argument("--max-search-trials", type=int, default=5)
    parser.add_argument("--random-search-seed", type=int, default=20260429)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--stall-seconds", type=int, default=None)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--variants", nargs="*", choices=sorted(ABLATION_FLAGS), default=list(ABLATION_FLAGS))
    parser.add_argument(
        "--feature-source-variants",
        nargs="*",
        choices=sorted(FEATURE_SOURCE_FLAGS),
        default=list(FEATURE_SOURCE_FLAGS),
        help="MC-TRCM feature-source restrictions to run after validation-selected K/hyperparameters are fixed.",
    )
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="core_v1",
        help="Prefix for generated run names. Use a distinct prefix for pilot runs.",
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default=None,
        help="Directory for protocol summaries. Defaults to results/mctrcm_core_protocol_v1.",
    )
    parser.add_argument(
        "--allow-incomplete-search-selection",
        action="store_true",
        help=(
            "Allow selected_hyperparameters.csv to be written from a partially "
            "completed search grid. By default a dataset is selected only after "
            "all expected K x candidate x seed runs have valid metrics."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def _result_root(args: argparse.Namespace) -> Path:
    if args.result_root is None or not str(args.result_root).strip():
        return RESULT_ROOT
    path = Path(str(args.result_root))
    if not path.is_absolute():
        path = ROOT / path
    return path


def _ensure_dirs(args: argparse.Namespace) -> None:
    _result_root(args).mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _dataset_base_command(dataset: str, python_exe: str) -> list[str]:
    command = [
        python_exe,
        "-m",
        "src.models.train_mctrcm_v2",
        "--datasets",
        dataset,
        "--model-config-path",
        "configs/model_configs/mctrcm_final.json",
        "--train-config-path",
        "configs/train_configs/mctrcm_v2_family_a.json",
        "--model-family-name",
        "mctrcm_core_v1",
        "--disable-concept-bottleneck",
        "--train-task-keys",
        *CORE_TASK_KEYS[dataset],
    ]
    if dataset == "deprest_cat":
        command.extend(
            [
                "--ordinal-labeldist-task-keys",
                "deprest_cat::gad7_cat",
                "deprest_cat::phq9_cat",
                "--ordinal-labeldist-weight",
                "0.25",
            ]
        )
    if dataset == "psyche_d":
        command.extend(
            [
                "--distill-mode",
                "strongest_clean",
                "--enable-psyche-two-stage",
                "--enable-psyche-binary-correction",
                "--enable-psyche-native-adapter",
                "--enable-psyche-taskwise-native-adapter",
                "--enable-psyche-delta-bridge",
            ]
        )
    return command


def _append_candidate_args(command: list[str], candidate: dict[str, object]) -> None:
    scalar_flags = {
        "learning_rate": "--learning-rate",
        "weight_decay": "--weight-decay",
        "dropout": "--dropout",
        "token_dim": "--token-dim",
        "latent_dim": "--latent-dim",
        "output_refine_dim": "--output-refine-dim",
        "batch_size": "--batch-size",
        "patience": "--patience",
        "regression_loss": "--regression-loss",
        "film_conditioning_mode": "--film-conditioning-mode",
        "conditioning_mode": "--conditioning-mode",
        "modality_dropout": "--modality-dropout",
        "class_balance_mode": "--class-balance-mode",
    }
    for key, flag in scalar_flags.items():
        value = candidate.get(key)
        if value is not None:
            command.extend([flag, str(value)])
    if candidate.get("missingness_embedding_norm"):
        command.append("--missingness-embedding-norm")
    if candidate.get("binary_focal_gamma") is not None:
        command.extend(["--binary-focal-task-keys", "psyche_d::phq_change_binary"])
        command.extend(["--binary-focal-gamma", str(candidate["binary_focal_gamma"])])
    if candidate.get("multiclass_focal_gamma") is not None:
        command.extend(["--multiclass-focal-task-keys", "psyche_d::phq_change_multiclass"])
        command.extend(["--multiclass-focal-gamma", str(candidate["multiclass_focal_gamma"])])
    if candidate.get("multiclass_label_smoothing") is not None:
        command.extend(["--multiclass-label-smoothing-task-keys", "psyche_d::phq_change_multiclass"])
        command.extend(["--multiclass-label-smoothing", str(candidate["multiclass_label_smoothing"])])


def _search_candidates(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.search_profile == "compact":
        return [dict(candidate) for candidate in COMPACT_CANDIDATES[: max(args.max_search_trials, 1)]]

    keys = list(SEARCH_SPACE)
    values = [SEARCH_SPACE[key] for key in keys]
    if args.search_profile == "full_grid":
        candidates = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        return candidates[: args.max_search_trials] if args.max_search_trials > 0 else candidates

    all_candidates = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    rng = random.Random(args.random_search_seed)
    rng.shuffle(all_candidates)
    return all_candidates[: max(args.max_search_trials, 1)]


def _metrics_path(run_name: str) -> Path:
    return PREDICTION_DIR / f"{run_name}__metrics.csv"


def _record_failed_run(
    args: argparse.Namespace,
    *,
    run_name: str,
    command: list[str],
    return_code: int,
    attempt: int,
) -> None:
    result_root = _result_root(args)
    path = result_root / "failed_runs.csv"
    row = pd.DataFrame(
        [
            {
                "run_name": run_name,
                "return_code": return_code,
                "attempt": attempt,
                "command": " ".join(command),
            }
        ]
    )
    if path.exists():
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, index=False)


def _run(command: list[str], *, run_name: str, args: argparse.Namespace) -> bool:
    metric_path = _metrics_path(run_name)
    if args.reuse_existing and metric_path.exists():
        print(f"[reuse] {run_name}")
        return True
    print(" ".join(command), flush=True)
    if args.dry_run:
        return True
    max_attempts = max(int(args.retries), 0) + 1
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(command, cwd=BUNDLE_ROOT, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            _record_failed_run(
                args,
                run_name=run_name,
                command=command,
                return_code=int(exc.returncode),
                attempt=attempt,
            )
            if attempt >= max_attempts:
                if args.continue_on_error:
                    print(f"[failed] {run_name} return_code={exc.returncode}", flush=True)
                    return False
                raise
            print(f"[retry] {run_name} attempt {attempt + 1}/{max_attempts}", flush=True)
    return False


def _command_for_run(
    *,
    dataset: str,
    run_name: str,
    seed: int,
    k: int,
    candidate: dict[str, object],
    args: argparse.Namespace,
    extra_flags: tuple[str, ...] = (),
    epochs: int | None = None,
) -> list[str]:
    command = _dataset_base_command(dataset, args.python_exe)
    command.extend(
        [
            "--run-name",
            run_name,
            "--seed",
            str(seed),
            "--recursion-steps",
            str(k),
            "--progress-log-path",
            f"outputs/logs/{run_name}.progress.log",
        ]
    )
    if args.stall_seconds is not None:
        command.extend(["--stall-seconds", str(args.stall_seconds)])
    _append_candidate_args(command, candidate)
    if epochs is not None:
        command.extend(["--epochs", str(epochs), "--min-epochs", "1", "--patience", "1"])
    elif args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    command.extend(extra_flags)
    return command


def run_smoke(args: argparse.Namespace) -> None:
    candidate = dict(COMPACT_CANDIDATES[0])
    for dataset in args.datasets[:1]:
        run_name = f"{args.run_prefix}_smoke__{dataset}__k1__seed{args.search_seeds[0]}"
        command = _command_for_run(
            dataset=dataset,
            run_name=run_name,
            seed=args.search_seeds[0],
            k=1,
            candidate=candidate,
            args=args,
            epochs=args.smoke_epochs,
        )
        _run(command, run_name=run_name, args=args)


def run_k_search(args: argparse.Namespace) -> None:
    candidates = _search_candidates(args)
    result_root = _result_root(args)
    (result_root / "search_candidates.json").write_text(
        json.dumps(candidates, indent=2),
        encoding="utf-8",
    )
    for dataset in args.datasets:
        for hp_index, candidate in enumerate(candidates):
            for k in args.k_values:
                for seed in args.search_seeds:
                    run_name = f"{args.run_prefix}_search__{dataset}__hp{hp_index:03d}__k{k}__seed{seed}"
                    command = _command_for_run(
                        dataset=dataset,
                        run_name=run_name,
                        seed=seed,
                        k=k,
                        candidate=candidate,
                        args=args,
                        extra_flags=("--skip-test-eval",),
                    )
                    _run(command, run_name=run_name, args=args)


def _primary_metric(task_key: str) -> str:
    return "r2" if task_key in REGRESSION_TASKS else "balanced_accuracy"


def _metric_value(row: pd.Series, task_key: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(_primary_metric(task_key))]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else math.nan


def _load_run_metrics(run_name: str) -> pd.DataFrame:
    path = _metrics_path(run_name)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, keep_default_na=False)
    frame["run_name"] = run_name
    frame["task_key"] = frame["dataset_id"].astype(str) + "::" + frame["task_name"].astype(str)
    frame = frame.loc[frame["task_key"].isin(TASK_DISPLAY)].copy()
    frame["primary_metric"] = frame["task_key"].map(_primary_metric)
    frame["primary_value"] = [
        _metric_value(row, str(row["task_key"]))
        for _, row in frame.iterrows()
    ]
    return frame


def _search_run_names(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    candidates = _search_candidates(args)
    for dataset in args.datasets:
        for hp_index, _candidate in enumerate(candidates):
            for k in args.k_values:
                for seed in args.search_seeds:
                    rows.append(
                        {
                            "dataset": dataset,
                            "hp_index": hp_index,
                            "k": k,
                            "seed": seed,
                            "run_name": f"{args.run_prefix}_search__{dataset}__hp{hp_index:03d}__k{k}__seed{seed}",
                        }
                    )
    return rows


def _write_search_completion(args: argparse.Namespace, frame: pd.DataFrame) -> set[str]:
    result_root = _result_root(args)
    expected = pd.DataFrame(_search_run_names(args))
    if expected.empty:
        return set()
    expected["expected_valid_tasks"] = expected["dataset"].map(lambda dataset: len(CORE_TASKS[str(dataset)]))
    expected["metrics_path"] = expected["run_name"].map(lambda run_name: str(_metrics_path(str(run_name))))

    if frame.empty:
        observed = pd.DataFrame(columns=["run_name", "observed_valid_tasks"])
    else:
        observed = (
            frame.loc[frame["split"].astype(str).eq("valid")]
            .groupby("run_name", as_index=False)["task_key"]
            .nunique()
            .rename(columns={"task_key": "observed_valid_tasks"})
        )
    completion = expected.merge(observed, on="run_name", how="left")
    completion["observed_valid_tasks"] = (
        pd.to_numeric(completion["observed_valid_tasks"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    completion["complete"] = completion["observed_valid_tasks"] >= completion["expected_valid_tasks"]
    completion.to_csv(result_root / "search_run_completion.csv", index=False)
    completion.loc[~completion["complete"]].to_csv(result_root / "missing_search_runs.csv", index=False)
    dataset_completion = completion.groupby("dataset")["complete"].all()
    return {str(dataset) for dataset, complete in dataset_completion.items() if bool(complete)}


def summarize_k_search(args: argparse.Namespace) -> pd.DataFrame:
    result_root = _result_root(args)
    rows = []
    for spec in _search_run_names(args):
        metrics = _load_run_metrics(str(spec["run_name"]))
        if metrics.empty:
            continue
        valid = metrics.loc[metrics["split"].astype(str).eq("valid")].copy()
        test = metrics.loc[metrics["split"].astype(str).eq("test")].copy()
        for split_name, split_frame in (("valid", valid), ("test", test)):
            for _, row in split_frame.iterrows():
                rows.append(
                    {
                        **spec,
                        "split": split_name,
                        "task_key": row["task_key"],
                        "task_display": TASK_DISPLAY[str(row["task_key"])],
                        "primary_metric": row["primary_metric"],
                        "primary_value": row["primary_value"],
                    }
                )
    frame = pd.DataFrame(rows)
    complete_datasets = _write_search_completion(args, frame)
    if frame.empty:
        frame.to_csv(result_root / "k_search_task_metrics.csv", index=False)
        empty_selected = pd.DataFrame(
            columns=["dataset", "hp_index", "k", "mean_valid_primary", "candidate_json"]
        )
        empty_selected.to_csv(result_root / "selected_hyperparameters.csv", index=False)
        (result_root / "selected_hyperparameters.json").write_text("[]", encoding="utf-8")
        return empty_selected
    for column in ("primary_value", "brier_score", "ece"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.to_csv(result_root / "k_search_task_metrics.csv", index=False)
    summary = (
        frame.loc[frame["split"].eq("valid")]
        .groupby(["dataset", "hp_index", "k"], as_index=False)["primary_value"]
        .mean()
        .rename(columns={"primary_value": "mean_valid_primary"})
        .sort_values(["dataset", "mean_valid_primary"], ascending=[True, False])
    )
    candidates = _search_candidates(args)
    summary["candidate_json"] = summary["hp_index"].map(lambda idx: json.dumps(candidates[int(idx)], sort_keys=True))
    summary.to_csv(result_root / "k_search_summary.csv", index=False)

    selection_frame = frame
    if not bool(getattr(args, "allow_incomplete_search_selection", False)):
        selection_frame = frame.loc[frame["dataset"].astype(str).isin(complete_datasets)].copy()
    if selection_frame.empty:
        selected = pd.DataFrame(
            columns=["dataset", "hp_index", "k", "mean_valid_primary", "candidate_json"]
        )
        selected.to_csv(result_root / "selected_hyperparameters.csv", index=False)
        (result_root / "selected_hyperparameters.json").write_text("[]", encoding="utf-8")
        return selected
    selection_summary = (
        selection_frame.loc[selection_frame["split"].eq("valid")]
        .groupby(["dataset", "hp_index", "k"], as_index=False)["primary_value"]
        .mean()
        .rename(columns={"primary_value": "mean_valid_primary"})
        .sort_values(["dataset", "mean_valid_primary"], ascending=[True, False])
    )
    selection_summary["candidate_json"] = selection_summary["hp_index"].map(
        lambda idx: json.dumps(candidates[int(idx)], sort_keys=True)
    )
    selected_rows = []
    for dataset, dataset_frame in selection_summary.groupby("dataset", sort=True):
        selected_rows.append(dataset_frame.iloc[0].to_dict())
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(result_root / "selected_hyperparameters.csv", index=False)
    (result_root / "selected_hyperparameters.json").write_text(
        selected.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    return selected


def _load_selected(args: argparse.Namespace) -> pd.DataFrame:
    path = _result_root(args) / "selected_hyperparameters.csv"
    if path.exists():
        return pd.read_csv(path)
    return summarize_k_search(args)


def run_final(args: argparse.Namespace) -> None:
    selected = _load_selected(args)
    if selected.empty:
        raise RuntimeError("No selected hyperparameters found. Run --stage k_search first.")
    candidates = _search_candidates(args)
    for row in selected.to_dict(orient="records"):
        dataset = str(row["dataset"])
        if dataset not in args.datasets:
            continue
        hp_index = int(row["hp_index"])
        k = int(row["k"])
        candidate = candidates[hp_index]
        for seed in args.final_seeds:
            run_name = f"{args.run_prefix}_final__{dataset}__hp{hp_index:03d}__k{k}__seed{seed}"
            command = _command_for_run(
                dataset=dataset,
                run_name=run_name,
                seed=seed,
                k=k,
                candidate=candidate,
                args=args,
            )
            _run(command, run_name=run_name, args=args)


def run_ablations(args: argparse.Namespace) -> None:
    selected = _load_selected(args)
    if selected.empty:
        raise RuntimeError("No selected hyperparameters found. Run --stage k_search first.")
    candidates = _search_candidates(args)
    for row in selected.to_dict(orient="records"):
        dataset = str(row["dataset"])
        if dataset not in args.datasets:
            continue
        hp_index = int(row["hp_index"])
        selected_k = int(row["k"])
        candidate = candidates[hp_index]
        for variant in args.variants:
            if variant == "full":
                print(
                    f"[reuse-final-as-full-ablation] {dataset} hp{hp_index:03d} k{selected_k}",
                    flush=True,
                )
                continue
            extra_flags = tuple(ABLATION_FLAGS[variant])
            k = 1 if variant == "no_recursive_refinement" else selected_k
            for seed in args.final_seeds:
                run_name = f"{args.run_prefix}_ablation__{variant}__{dataset}__hp{hp_index:03d}__k{k}__seed{seed}"
                command = _command_for_run(
                    dataset=dataset,
                    run_name=run_name,
                    seed=seed,
                    k=k,
                    candidate=candidate,
                    args=args,
                    extra_flags=extra_flags,
                )
                _run(command, run_name=run_name, args=args)


def run_feature_sources(args: argparse.Namespace) -> None:
    selected = _load_selected(args)
    if selected.empty:
        raise RuntimeError("No selected hyperparameters found. Run --stage k_search first.")
    candidates = _search_candidates(args)
    for row in selected.to_dict(orient="records"):
        dataset = str(row["dataset"])
        if dataset not in args.datasets:
            continue
        hp_index = int(row["hp_index"])
        selected_k = int(row["k"])
        candidate = candidates[hp_index]
        for variant in args.feature_source_variants:
            if variant == "full":
                print(
                    f"[reuse-final-as-feature-source-full] {dataset} hp{hp_index:03d} k{selected_k}",
                    flush=True,
                )
                continue
            extra_flags = tuple(FEATURE_SOURCE_FLAGS[variant])
            for seed in args.final_seeds:
                run_name = (
                    f"{args.run_prefix}_feature_source__{variant}__"
                    f"{dataset}__hp{hp_index:03d}__k{selected_k}__seed{seed}"
                )
                command = _command_for_run(
                    dataset=dataset,
                    run_name=run_name,
                    seed=seed,
                    k=selected_k,
                    candidate=candidate,
                    args=args,
                    extra_flags=extra_flags,
                )
                _run(command, run_name=run_name, args=args)


def _full_feature_summary() -> pd.DataFrame:
    path = ROOT / "results" / "revised_fairness_v1" / "summary_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, keep_default_na=False)
    frame = frame.loc[
        frame["feature_set"].astype(str).eq("FULL")
        & frame["dataset"].astype(str).isin(CORE_TASKS)
        & frame["model_family"].astype(str).ne("")
    ].copy()
    frame["task_key"] = frame["dataset"].astype(str) + "::" + frame["endpoint"].astype(str)
    frame = frame.loc[frame["task_key"].isin(TASK_DISPLAY)].copy()
    for column in ("mean_val_primary", "mean_test_primary", "se_test_primary"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _selected_family_table(
    frame: pd.DataFrame,
    *,
    families: set[str],
    prefix: str,
    allow_mctrcm: bool = False,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    subset = frame.loc[frame["model_family"].astype(str).isin(families)].copy()
    if not allow_mctrcm:
        subset = subset.loc[~subset["model_family"].astype(str).str.contains("mctrcm", case=False, na=False)].copy()
    if subset.empty:
        return subset
    idx = subset.groupby("task_key")["mean_val_primary"].idxmax()
    selected = subset.loc[idx].copy()
    return selected.rename(
        columns={
            "model_family": f"{prefix}_family",
            "mean_val_primary": f"{prefix}_mean_val_primary",
            "mean_test_primary": f"{prefix}_mean_test_primary",
            "se_test_primary": f"{prefix}_se_test_primary",
        }
    )


def _overall_baseline_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    baseline_families = TABULAR_BASELINE_FAMILIES | NEURAL_BASELINE_FAMILIES
    subset = frame.loc[frame["model_family"].astype(str).isin(baseline_families)].copy()
    if subset.empty:
        return subset
    idx = subset.groupby("task_key")["mean_val_primary"].idxmax()
    return subset.loc[idx].copy().rename(
        columns={
            "model_family": "best_baseline",
            "mean_val_primary": "baseline_mean_val_primary",
            "mean_test_primary": "baseline_mean_test_primary",
            "se_test_primary": "baseline_se_test_primary",
        }
    )


def _add_endpoint_ranks(frame: pd.DataFrame, result_root: Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    value_columns = {
        "null_rank": "null_mean_test_primary",
        "tabular_rank": "tabular_mean_test_primary",
        "neural_rank": "neural_mean_test_primary",
        "mctrcm_rank": "mctrcm_mean_primary",
    }
    ranked = frame.copy()
    for row_index, row in ranked.iterrows():
        values = {
            label: pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            for label, column in value_columns.items()
        }
        valid = pd.Series({label: value for label, value in values.items() if pd.notna(value)})
        if valid.empty:
            continue
        ranks = valid.rank(ascending=False, method="min")
        for label, rank_value in ranks.items():
            ranked.loc[row_index, label] = float(rank_value)
    rank_rows = []
    for label in value_columns:
        mean_rank = pd.to_numeric(ranked.get(label), errors="coerce").mean()
        rank_rows.append({"method": label.removesuffix("_rank"), "average_rank": mean_rank})
        ranked[f"{label.removesuffix('_rank')}_average_rank"] = mean_rank
    pd.DataFrame(rank_rows).to_csv(result_root / "core_average_ranks.csv", index=False)
    return ranked


def summarize_final(args: argparse.Namespace) -> pd.DataFrame:
    selected = _load_selected(args)
    rows = []
    for row in selected.to_dict(orient="records"):
        dataset = str(row["dataset"])
        if dataset not in args.datasets:
            continue
        hp_index = int(row["hp_index"])
        k = int(row["k"])
        for seed in args.final_seeds:
            run_name = f"{args.run_prefix}_final__{dataset}__hp{hp_index:03d}__k{k}__seed{seed}"
            metrics = _load_run_metrics(run_name)
            if metrics.empty:
                continue
            for _, metric_row in metrics.iterrows():
                rows.append(
                    {
                        "run_name": run_name,
                        "dataset": dataset,
                        "seed": seed,
                        "hp_index": hp_index,
                        "validation_selected_k": k,
                        "split": metric_row["split"],
                        "task_key": metric_row["task_key"],
                        "task_display": TASK_DISPLAY[str(metric_row["task_key"])],
                        "primary_metric": metric_row["primary_metric"],
                        "primary_value": metric_row["primary_value"],
                        "r2": metric_row.get("r2"),
                        "rmse": metric_row.get("rmse"),
                        "mae": metric_row.get("mae"),
                        "spearman": metric_row.get("spearman"),
                        "balanced_accuracy": metric_row.get("balanced_accuracy"),
                        "macro_f1": metric_row.get("macro_f1"),
                        "auroc": metric_row.get("auroc"),
                        "auprc": metric_row.get("auprc"),
                        "brier_score": metric_row.get("brier_score"),
                        "ece": metric_row.get("ece"),
                    }
                )
    seed_metrics = pd.DataFrame(rows)
    if seed_metrics.empty:
        return seed_metrics
    numeric_columns = [
        "primary_value",
        "r2",
        "rmse",
        "mae",
        "spearman",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "auprc",
        "brier_score",
        "ece",
    ]
    for column in numeric_columns:
        if column in seed_metrics.columns:
            seed_metrics[column] = pd.to_numeric(seed_metrics[column], errors="coerce")
    result_root = _result_root(args)
    seed_metrics.to_csv(result_root / "final_seed_metrics.csv", index=False)
    test = seed_metrics.loc[seed_metrics["split"].astype(str).eq("test")].copy()
    aggregate = (
        test.groupby(["dataset", "task_key", "task_display", "primary_metric", "validation_selected_k"], as_index=False)
        .agg(
            mctrcm_mean_primary=("primary_value", "mean"),
            mctrcm_se_primary=("primary_value", lambda values: float(pd.Series(values).std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0),
            n_seeds=("seed", "nunique"),
            mctrcm_mean_brier=("brier_score", "mean"),
            mctrcm_mean_ece=("ece", "mean"),
        )
    )
    summary_frame = _full_feature_summary()
    selected_tables = [
        _selected_family_table(summary_frame, families=NULL_BASELINE_FAMILIES, prefix="null"),
        _selected_family_table(summary_frame, families=TABULAR_BASELINE_FAMILIES, prefix="tabular"),
        _selected_family_table(summary_frame, families=NEURAL_BASELINE_FAMILIES, prefix="neural"),
        _overall_baseline_table(summary_frame),
    ]
    for selected_table in selected_tables:
        if selected_table.empty:
            continue
        keep_columns = [
            column
            for column in selected_table.columns
            if column == "task_key"
            or column.endswith("_family")
            or column.endswith("_mean_val_primary")
            or column.endswith("_mean_test_primary")
            or column.endswith("_se_test_primary")
            or column == "best_baseline"
            or column.startswith("baseline_")
        ]
        aggregate = aggregate.merge(selected_table[keep_columns], on="task_key", how="left")
    if "baseline_mean_test_primary" in aggregate.columns:
        aggregate["delta_vs_best_baseline"] = (
            aggregate["mctrcm_mean_primary"] - aggregate["baseline_mean_test_primary"]
        )
    aggregate = _add_endpoint_ranks(aggregate, result_root)
    aggregate.to_csv(result_root / "core_main_comparison.csv", index=False)
    _write_latex_main_table(aggregate, result_root)
    return aggregate


def _fmt(value: object) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(value):
        return "--"
    return f"{value:.3f}"


def _write_latex_main_table(frame: pd.DataFrame, result_root: Path) -> None:
    if frame.empty:
        return
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Core mobile and wearable depression feature-view benchmark under validation-selected recursive depth.}",
        r"\label{tab:core_main_comparison}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Endpoint & Metric & Null & Validation-selected tabular reference & Validation-selected neural reference & MC-TRCM & $\Delta$ vs stronger displayed validation reference & K & Rank \\",
        r"\midrule",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            f"{row['task_display']} & {row['primary_metric']} & "
            f"{_fmt(row.get('null_mean_test_primary'))} & "
            f"{row.get('tabular_family', '--')} {_fmt(row.get('tabular_mean_test_primary'))} & "
            f"{row.get('neural_family', '--')} {_fmt(row.get('neural_mean_test_primary'))} & "
            f"{_fmt(row.get('mctrcm_mean_primary'))} & "
            f"{_fmt(row.get('delta_vs_best_baseline'))} & "
            f"{int(row.get('validation_selected_k', 0))} & "
            f"{_fmt(row.get('mctrcm_rank'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (result_root / "core_main_comparison.tex").write_text("\n".join(lines), encoding="utf-8")


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if str(column).startswith("proba_")]
    valid_columns = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            valid_columns.append(column)
    return sorted(valid_columns, key=lambda column: int(str(column).split("_", 1)[1]))


def _probability_matrix(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    proba_columns = _probability_columns(frame)
    if not proba_columns:
        return proba_columns, np.empty((len(frame), 0), dtype=float)
    matrix = frame[proba_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, 1e-8, 1.0)
    matrix = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1e-8)
    return proba_columns, matrix


def _macro_f1_for_indices(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), average="macro"))


def _balanced_accuracy_for_indices(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)))


def _fit_binary_threshold(valid_task: pd.DataFrame) -> float:
    _columns, probabilities = _probability_matrix(valid_task)
    if probabilities.shape[1] < 2:
        return 0.5
    y_true = pd.to_numeric(valid_task["y_true_index"], errors="coerce").fillna(0).to_numpy(dtype=int)
    scores = probabilities[:, 1]
    candidates = sorted(set(float(value) for value in scores if math.isfinite(float(value))))
    candidates.extend(float(value) for value in np.linspace(0.05, 0.95, 19))
    best = (float("-inf"), float("-inf"), float("-inf"), 0.5)
    for threshold in candidates:
        y_pred = (scores >= threshold).astype(int)
        ba = _balanced_accuracy_for_indices(y_true, y_pred)
        macro_f1 = _macro_f1_for_indices(y_true, y_pred)
        tie_break = -abs(float(threshold) - 0.5)
        best = max(best, (ba, macro_f1, tie_break, float(threshold)))
    return best[3]


def _fit_class_bias(valid_task: pd.DataFrame) -> np.ndarray:
    _columns, probabilities = _probability_matrix(valid_task)
    if probabilities.shape[1] <= 1:
        return np.zeros(probabilities.shape[1], dtype=float)
    y_true = pd.to_numeric(valid_task["y_true_index"], errors="coerce").fillna(0).to_numpy(dtype=int)
    grid = (-1.0, -0.5, 0.0, 0.5, 1.0)
    best_score = (float("-inf"), float("-inf"))
    best_bias = np.zeros(probabilities.shape[1], dtype=float)
    import itertools

    for tail_bias in itertools.product(grid, repeat=probabilities.shape[1] - 1):
        bias = np.asarray((0.0, *tail_bias), dtype=float)
        logits = np.log(np.clip(probabilities, 1e-8, 1.0)) + bias
        y_pred = logits.argmax(axis=1)
        score = (_balanced_accuracy_for_indices(y_true, y_pred), _macro_f1_for_indices(y_true, y_pred))
        if score > best_score:
            best_score = score
            best_bias = bias
    return best_bias


def _apply_class_bias(probabilities: np.ndarray, bias: np.ndarray) -> np.ndarray:
    if probabilities.size == 0:
        return probabilities
    logits = np.log(np.clip(probabilities, 1e-8, 1.0)) + bias.reshape(1, -1)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True).clip(min=1e-8)


def _classification_metric_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    n_classes = max(
        int(np.nanmax(y_true)) + 1 if y_true.size else 1,
        int(np.nanmax(y_pred)) + 1 if y_pred.size else 1,
        probabilities.shape[1],
    )
    if probabilities.shape[1] < n_classes:
        padded = np.zeros((probabilities.shape[0], n_classes), dtype=float)
        padded[:, : probabilities.shape[1]] = probabilities
        probabilities = padded
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True).clip(min=1e-8)

    confidences = probabilities.max(axis=1)
    correctness = (probabilities.argmax(axis=1) == y_true).astype(float)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lower) & (confidences <= upper if upper == 1.0 else confidences < upper)
        if np.any(mask):
            ece += float(mask.mean() * abs(correctness[mask].mean() - confidences[mask].mean()))

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "auroc": math.nan,
        "auprc": math.nan,
        "brier_score": math.nan,
        "ece": ece,
    }
    unique_classes = np.unique(y_true)
    if n_classes == 2:
        if unique_classes.size >= 2:
            metrics["auroc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
            metrics["auprc"] = float(average_precision_score(y_true, probabilities[:, 1]))
        metrics["brier_score"] = float(brier_score_loss(y_true, probabilities[:, 1]))
        return metrics

    y_true_one_hot = label_binarize(y_true, classes=np.arange(n_classes))
    valid_columns = y_true_one_hot.sum(axis=0) > 0
    if valid_columns.sum() >= 2:
        metrics["auroc"] = float(
            roc_auc_score(
                y_true_one_hot[:, valid_columns],
                probabilities[:, valid_columns],
                multi_class="ovr",
                average="macro",
            )
        )
    if valid_columns.sum() >= 1:
        metrics["auprc"] = float(
            average_precision_score(
                y_true_one_hot[:, valid_columns],
                probabilities[:, valid_columns],
                average="macro",
            )
        )
    metrics["brier_score"] = float(np.mean(np.sum((y_true_one_hot - probabilities) ** 2, axis=1)))
    return metrics


def _regression_metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
    }


def _ensemble_frame(seed_frames: list[pd.DataFrame]) -> pd.DataFrame:
    id_columns = ["split", "dataset_id", "subject_id", "anchor_id", "task_name", "task_key", "label_type"]
    rows = []
    for seed_index, frame in enumerate(seed_frames):
        local = frame.copy()
        local["ensemble_seed_index"] = seed_index
        rows.append(local)
    stacked = pd.concat(rows, ignore_index=True)
    proba_columns = _probability_columns(stacked)
    value_columns = ["y_true", "y_true_index", "y_pred", "y_pred_index", *proba_columns]
    for column in value_columns:
        if column in stacked.columns:
            stacked[column] = pd.to_numeric(stacked[column], errors="coerce")

    output_rows = []
    for keys, group in stacked.groupby(id_columns, dropna=False, sort=False):
        row = dict(zip(id_columns, keys))
        first = group.iloc[0]
        row["y_true"] = first.get("y_true")
        row["y_true_index"] = first.get("y_true_index")
        if str(row["label_type"]) == "continuous":
            row["y_pred"] = float(group["y_pred"].mean())
            row["y_pred_index"] = np.nan
        else:
            group_proba_columns = _probability_columns(group)
            probabilities = group[group_proba_columns].mean(axis=0).to_numpy(dtype=float)
            if probabilities.size == 0 or not np.isfinite(probabilities).any():
                probabilities = np.eye(int(group["y_pred_index"].max()) + 1)[group["y_pred_index"].astype(int)].mean(axis=0)
            probabilities = np.nan_to_num(probabilities, nan=0.0)
            total = float(probabilities.sum())
            if total <= 0.0:
                probabilities[:] = 1.0 / max(len(probabilities), 1)
            else:
                probabilities = probabilities / total
            y_pred_index = int(np.argmax(probabilities))
            row["y_pred_index"] = y_pred_index
            row["y_pred"] = float(y_pred_index)
            for column, value in zip(group_proba_columns, probabilities):
                row[column] = float(value)
        output_rows.append(row)
    return pd.DataFrame(output_rows)


def summarize_ensembles(args: argparse.Namespace) -> pd.DataFrame:
    selected = _load_selected(args)
    result_root = _result_root(args)
    completion_rows = []
    prediction_frames = []
    metric_rows = []
    for row in selected.to_dict(orient="records"):
        dataset = str(row["dataset"])
        if dataset not in args.datasets:
            continue
        hp_index = int(row["hp_index"])
        k = int(row["k"])
        valid_seed_frames = []
        test_seed_frames = []
        for seed in args.final_seeds:
            run_name = f"{args.run_prefix}_final__{dataset}__hp{hp_index:03d}__k{k}__seed{seed}"
            valid_path = PREDICTION_DIR / f"{run_name}__valid.csv"
            test_path = PREDICTION_DIR / f"{run_name}__test.csv"
            completion_rows.append(
                {
                    "dataset": dataset,
                    "run_name": run_name,
                    "valid_prediction_path": str(valid_path),
                    "test_prediction_path": str(test_path),
                    "exists": valid_path.exists() and test_path.exists(),
                }
            )
            if valid_path.exists() and test_path.exists():
                valid_seed_frames.append(pd.read_csv(valid_path, keep_default_na=False))
                test_seed_frames.append(pd.read_csv(test_path, keep_default_na=False))
        if len(valid_seed_frames) != len(args.final_seeds) or len(test_seed_frames) != len(args.final_seeds):
            continue
        valid_ensemble = _ensemble_frame(valid_seed_frames)
        test_ensemble = _ensemble_frame(test_seed_frames)
        test_ensemble["dataset"] = dataset
        test_ensemble["hp_index"] = hp_index
        test_ensemble["validation_selected_k"] = k
        test_ensemble["ensemble_calibration"] = "validation_selected"
        for (task_key, task_name, label_type), task_frame in test_ensemble.groupby(
            ["task_key", "task_name", "label_type"], dropna=False
        ):
            task_mask = test_ensemble["task_key"].astype(str).eq(str(task_key))
            valid_task = valid_ensemble.loc[valid_ensemble["task_key"].astype(str).eq(str(task_key))].copy()
            calibration_note = ""
            if str(label_type) == "continuous":
                metrics = _regression_metric_dict(task_frame["y_true"].to_numpy(), task_frame["y_pred"].to_numpy())
                primary_metric = "r2"
                primary_value = metrics["r2"]
                calibration_note = "mean_prediction"
            else:
                proba_columns, probabilities = _probability_matrix(task_frame)
                y_true = pd.to_numeric(task_frame["y_true_index"], errors="coerce").fillna(0).to_numpy(dtype=int)
                if str(label_type) == "binary" and probabilities.shape[1] >= 2 and not valid_task.empty:
                    threshold = _fit_binary_threshold(valid_task)
                    y_pred = (probabilities[:, 1] >= threshold).astype(int)
                    calibration_note = f"threshold={threshold:.6f}"
                elif probabilities.shape[1] >= 2 and not valid_task.empty:
                    bias = _fit_class_bias(valid_task)
                    probabilities = _apply_class_bias(probabilities, bias)
                    y_pred = probabilities.argmax(axis=1).astype(int)
                    calibration_note = "bias=" + json.dumps([float(value) for value in bias])
                    for column_index, column in enumerate(proba_columns):
                        test_ensemble.loc[task_mask, column] = probabilities[:, column_index]
                else:
                    y_pred = pd.to_numeric(task_frame["y_pred_index"], errors="coerce").fillna(0).to_numpy(dtype=int)
                    calibration_note = "argmax"
                test_ensemble.loc[task_mask, "y_pred_index"] = y_pred
                test_ensemble.loc[task_mask, "y_pred"] = y_pred.astype(float)
                metrics = _classification_metric_dict(
                    y_true,
                    y_pred,
                    probabilities,
                )
                primary_metric = "balanced_accuracy"
                primary_value = metrics["balanced_accuracy"]
            metric_rows.append(
                {
                    "dataset": dataset,
                    "task_key": task_key,
                    "task_name": task_name,
                    "task_display": TASK_DISPLAY.get(str(task_key), str(task_key)),
                    "label_type": label_type,
                    "validation_selected_k": k,
                    "n_seeds": len(test_seed_frames),
                    "primary_metric": primary_metric,
                    "primary_value": primary_value,
                    "ensemble_calibration": calibration_note,
                    **metrics,
                }
            )
        prediction_frames.append(test_ensemble)
    pd.DataFrame(completion_rows).to_csv(result_root / "ensemble_completion.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    predictions.to_csv(result_root / "mctrcm_seed_ensemble_predictions.csv", index=False)
    metrics.to_csv(result_root / "mctrcm_seed_ensemble_metrics.csv", index=False)
    return metrics


def summarize_ablations(args: argparse.Namespace) -> pd.DataFrame:
    selected = _load_selected(args)
    rows = []
    completion_rows = []
    for row in selected.to_dict(orient="records"):
        dataset = str(row["dataset"])
        if dataset not in args.datasets:
            continue
        hp_index = int(row["hp_index"])
        selected_k = int(row["k"])
        for variant in args.variants:
            k = 1 if variant == "no_recursive_refinement" else selected_k
            for seed in args.final_seeds:
                if variant == "full":
                    run_name = f"{args.run_prefix}_final__{dataset}__hp{hp_index:03d}__k{selected_k}__seed{seed}"
                else:
                    run_name = f"{args.run_prefix}_ablation__{variant}__{dataset}__hp{hp_index:03d}__k{k}__seed{seed}"
                metrics = _load_run_metrics(run_name)
                test = metrics.loc[metrics["split"].astype(str).eq("test")].copy() if not metrics.empty else pd.DataFrame()
                observed_test_tasks = int(test["task_key"].nunique()) if not test.empty else 0
                completion_rows.append(
                    {
                        "variant": variant,
                        "dataset": dataset,
                        "seed": seed,
                        "hp_index": hp_index,
                        "k": k,
                        "run_name": run_name,
                        "expected_test_tasks": len(CORE_TASKS[dataset]),
                        "observed_test_tasks": observed_test_tasks,
                        "metrics_path": str(_metrics_path(run_name)),
                        "complete": observed_test_tasks >= len(CORE_TASKS[dataset]),
                    }
                )
                if metrics.empty:
                    continue
                for _, metric_row in test.iterrows():
                    rows.append(
                        {
                            "variant": variant,
                            "dataset": dataset,
                            "seed": seed,
                            "hp_index": hp_index,
                            "k": k,
                            "task_key": metric_row["task_key"],
                            "task_display": TASK_DISPLAY[str(metric_row["task_key"])],
                            "primary_metric": metric_row["primary_metric"],
                            "primary_value": metric_row["primary_value"],
                            "brier_score": metric_row.get("brier_score"),
                            "ece": metric_row.get("ece"),
                        }
                    )
    frame = pd.DataFrame(rows)
    result_root = _result_root(args)
    completion = pd.DataFrame(completion_rows)
    completion.to_csv(result_root / "ablation_run_completion.csv", index=False)
    if not completion.empty:
        completion.loc[~completion["complete"]].to_csv(result_root / "missing_ablation_runs.csv", index=False)
    else:
        completion.to_csv(result_root / "missing_ablation_runs.csv", index=False)
    if frame.empty:
        return frame
    for column in ("primary_value", "brier_score", "ece"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.to_csv(result_root / "ablation_seed_metrics.csv", index=False)
    summary = (
        frame.groupby(["variant", "dataset", "task_key", "task_display", "primary_metric"], as_index=False)
        .agg(
            mean_primary=("primary_value", "mean"),
            se_primary=("primary_value", lambda values: float(pd.Series(values).std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0),
            n_seeds=("seed", "nunique"),
            mean_brier=("brier_score", "mean"),
            mean_ece=("ece", "mean"),
        )
    )
    full = summary.loc[summary["variant"].eq("full"), ["task_key", "mean_primary"]].rename(
        columns={"mean_primary": "full_mean_primary"}
    )
    summary = summary.merge(full, on="task_key", how="left")
    summary["delta_vs_full"] = summary["mean_primary"] - summary["full_mean_primary"]
    summary.to_csv(result_root / "ablation_summary.csv", index=False)
    return summary


def summarize_feature_sources(args: argparse.Namespace) -> pd.DataFrame:
    selected = _load_selected(args)
    rows = []
    completion_rows = []
    for row in selected.to_dict(orient="records"):
        dataset = str(row["dataset"])
        if dataset not in args.datasets:
            continue
        hp_index = int(row["hp_index"])
        selected_k = int(row["k"])
        for variant in args.feature_source_variants:
            for seed in args.final_seeds:
                if variant == "full":
                    run_name = f"{args.run_prefix}_final__{dataset}__hp{hp_index:03d}__k{selected_k}__seed{seed}"
                else:
                    run_name = (
                        f"{args.run_prefix}_feature_source__{variant}__"
                        f"{dataset}__hp{hp_index:03d}__k{selected_k}__seed{seed}"
                    )
                metrics = _load_run_metrics(run_name)
                test = metrics.loc[metrics["split"].astype(str).eq("test")].copy() if not metrics.empty else pd.DataFrame()
                observed_test_tasks = int(test["task_key"].nunique()) if not test.empty else 0
                completion_rows.append(
                    {
                        "feature_source_variant": variant,
                        "dataset": dataset,
                        "seed": seed,
                        "hp_index": hp_index,
                        "k": selected_k,
                        "run_name": run_name,
                        "expected_test_tasks": len(CORE_TASKS[dataset]),
                        "observed_test_tasks": observed_test_tasks,
                        "metrics_path": str(_metrics_path(run_name)),
                        "complete": observed_test_tasks >= len(CORE_TASKS[dataset]),
                    }
                )
                if metrics.empty:
                    continue
                for _, metric_row in test.iterrows():
                    rows.append(
                        {
                            "feature_source_variant": variant,
                            "dataset": dataset,
                            "seed": seed,
                            "hp_index": hp_index,
                            "k": selected_k,
                            "task_key": metric_row["task_key"],
                            "task_display": TASK_DISPLAY[str(metric_row["task_key"])],
                            "primary_metric": metric_row["primary_metric"],
                            "primary_value": metric_row["primary_value"],
                            "brier_score": metric_row.get("brier_score"),
                            "ece": metric_row.get("ece"),
                        }
                    )
    result_root = _result_root(args)
    completion = pd.DataFrame(completion_rows)
    completion.to_csv(result_root / "feature_source_run_completion.csv", index=False)
    if not completion.empty:
        completion.loc[~completion["complete"]].to_csv(result_root / "missing_feature_source_runs.csv", index=False)
    else:
        completion.to_csv(result_root / "missing_feature_source_runs.csv", index=False)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame.to_csv(result_root / "feature_source_seed_metrics.csv", index=False)
        return frame
    for column in ("primary_value", "brier_score", "ece"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.to_csv(result_root / "feature_source_seed_metrics.csv", index=False)
    summary = (
        frame.groupby(
            ["feature_source_variant", "dataset", "task_key", "task_display", "primary_metric"],
            as_index=False,
        )
        .agg(
            mean_primary=("primary_value", "mean"),
            se_primary=("primary_value", lambda values: float(pd.Series(values).std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0),
            n_seeds=("seed", "nunique"),
            mean_brier=("brier_score", "mean"),
            mean_ece=("ece", "mean"),
        )
    )
    summary.to_csv(result_root / "feature_source_mctrcm_summary.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    _ensure_dirs(args)
    if args.stage in {"smoke", "all"}:
        run_smoke(args)
    if args.stage in {"k_search", "all"}:
        run_k_search(args)
    if args.stage in {"final", "all"}:
        run_final(args)
    if args.stage in {"ablations", "all"}:
        run_ablations(args)
    if args.stage in {"feature_sources", "all"}:
        run_feature_sources(args)
    if args.stage in {"ensemble", "all"}:
        summarize_ensembles(args)
    if args.stage in {"summarize", "all", "k_search", "final", "ablations", "feature_sources"}:
        summarize_k_search(args)
        summarize_final(args)
        summarize_ensembles(args)
        summarize_ablations(args)
        summarize_feature_sources(args)


if __name__ == "__main__":
    main()
