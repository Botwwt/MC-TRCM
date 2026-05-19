from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "mctrcm_core_protocol_v1"
BASELINE_ROOT = ROOT / "results" / "revised_fairness_v1"
TABLE_ROOT = ROOT / "tables" / "core_protocol"


CORE_TASK_ORDER = [
    "deprest_cat::phq9_reg",
    "deprest_cat::gad7_reg",
    "deprest_cat::phq9_cat",
    "deprest_cat::gad7_cat",
    "psyche_d::phq_change_binary",
    "psyche_d::phq_change_multiclass",
]
TASK_DISPLAY = {
    "deprest_cat::phq9_reg": "DepreST-CAT PHQ-9 severity",
    "deprest_cat::gad7_reg": "DepreST-CAT GAD-7 severity",
    "deprest_cat::phq9_cat": "DepreST-CAT PHQ-9 category",
    "deprest_cat::gad7_cat": "DepreST-CAT GAD-7 category",
    "psyche_d::phq_change_binary": "PSYCHE-D PHQ-change binary",
    "psyche_d::phq_change_multiclass": "PSYCHE-D PHQ-change multiclass",
}
DATASET_DISPLAY = {
    "studentlife": "StudentLife",
    "deprest_cat": "DepreST-CAT",
    "psyche_d": "PSYCHE-D",
    "depresjon": "Depresjon",
    "obf": "OBF-Psychiatric",
}
APPENDIX_TASK_DISPLAY = {
    "studentlife::phq9_reg": "StudentLife PHQ-9 severity",
    "studentlife::phq9_cat": "StudentLife PHQ-9 category",
    "deprest_cat::phq9_reg": "DepreST-CAT PHQ-9 severity",
    "deprest_cat::gad7_reg": "DepreST-CAT GAD-7 severity",
    "deprest_cat::phq9_cat": "DepreST-CAT PHQ-9 category",
    "deprest_cat::gad7_cat": "DepreST-CAT GAD-7 category",
    "psyche_d::phq_change_binary": "PSYCHE-D PHQ-change binary",
    "psyche_d::phq_change_multiclass": "PSYCHE-D PHQ-change multiclass",
    "depresjon::madrs_reg": "Depresjon MADRS severity",
    "depresjon::dep_binary": "Depresjon depression status",
    "obf::clinical_vs_control": "OBF clinical vs. control",
    "obf::dep_binary": "OBF depression status",
    "obf::obf_5class": "OBF five-class group",
}
FEATURE_SET_DISPLAY = {
    "FULL": "Full",
    "SENSOR_ONLY": "Sensor",
    "SENSOR_VALUES_ONLY": "Sensor values",
    "SENSOR_PLUS_STATIC": "Sensor+static",
    "MISSINGNESS_ONLY": "Missingness",
    "STATIC_CLINICAL_ONLY": "Static/clinical",
    "SYMPTOM_CONTEXT_ONLY": "Symptom",
    "SYMPTOM_STATIC_CLINICAL": "Symptom/static/clinical",
}
CLASSIFICATION_TASKS = {
    "deprest_cat::phq9_cat",
    "deprest_cat::gad7_cat",
    "psyche_d::phq_change_binary",
    "psyche_d::phq_change_multiclass",
}
MODEL_DISPLAY = {
    "null": "Null",
    "elastic_net": "Elastic Net",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "ebm": "EBM",
    "mlp": "MLP",
    "simple_multitask_mlp": "Multi-task MLP",
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "mctrcm": "MC-TRCM",
}

APPENDIX_BASELINE_ORDER = [
    "elastic_net",
    "lightgbm",
    "xgboost",
    "ebm",
    "mlp",
    "gru",
    "lstm",
    "transformer",
    "mctrcm",
    "simple_multitask_mlp",
    "null",
]


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    if abs(number) < 0.0005:
        number = 0.0
    return f"{number:.{digits}f}"


def _fmt_pm(mean_value: object, se_value: object, digits: int = 3) -> str:
    try:
        mean_number = float(mean_value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(mean_number):
        return "--"
    try:
        se_number = float(se_value)
    except (TypeError, ValueError):
        return _fmt(mean_number, digits=digits)
    if not math.isfinite(se_number):
        return _fmt(mean_number, digits=digits)
    return f"{_fmt(mean_number, digits=digits)} $\\pm$ {_fmt(se_number, digits=digits)}"


def _standard_error(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) <= 1:
        return 0.0
    return float(numeric.std(ddof=1) / math.sqrt(len(numeric)))


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _label_suffix(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _latex_table(path: Path, *, caption: str, label: str, columns: str, header: str, rows: list[str]) -> None:
    env = "table"
    lines = [
        f"\\begin{{{env}}}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.12}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        header,
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        f"\\end{{{env}}}",
    ]
    _write(path, lines)


def _latex_longtable(path: Path, *, caption: str, label: str, columns: str, header: str, rows: list[str], size: str = "\\scriptsize") -> None:
    lines = [
        "\\begingroup",
        size,
        "\\setlength{\\tabcolsep}{1pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        f"\\begin{{longtable}}{{{columns}}}",
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
        "\\toprule",
        header,
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        header,
        "\\midrule",
        "\\endhead",
        *rows,
        "\\bottomrule",
        "\\end{longtable}",
        "\\endgroup",
    ]
    _write(path, lines)


def _latex_table_blocks(
    path: Path,
    *,
    blocks: list[dict[str, object]],
    columns: str,
    header: str,
    size: str = "\\tiny",
    tabcolsep: str = "1pt",
    arraystretch: str = "1.02",
) -> None:
    lines: list[str] = []
    for block in blocks:
        label = str(block.get("label", "") or "")
        caption = str(block["caption"])
        rows = list(block["rows"])
        lines.extend(["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}"])
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.extend(
            [
                size,
                f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
                f"\\renewcommand{{\\arraystretch}}{{{arraystretch}}}",
                f"\\begin{{tabular}}{{{columns}}}",
                "\\toprule",
                header,
                "\\midrule",
                *rows,
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        )
    _write(path, lines)


def main_comparison() -> None:
    frame = pd.read_csv(RESULT_ROOT / "core_main_comparison.csv")
    ensemble_path = RESULT_ROOT / "mctrcm_seed_ensemble_metrics.csv"
    ensembles = pd.read_csv(ensemble_path) if ensemble_path.exists() else pd.DataFrame()
    ranks_path = RESULT_ROOT / "core_average_ranks.csv"
    ranks = pd.read_csv(ranks_path, keep_default_na=False) if ranks_path.exists() else pd.DataFrame()
    frame["task_key"] = pd.Categorical(frame["task_key"], categories=CORE_TASK_ORDER, ordered=True)
    frame = frame.sort_values("task_key")
    rows = []
    for _, row in frame.iterrows():
        task_key = str(row["task_key"])
        ensemble = "--"
        if not ensembles.empty:
            erow = ensembles.loc[ensembles["task_key"].eq(task_key)]
            if not erow.empty:
                ensemble = _fmt(erow.iloc[0]["primary_value"])
        tabular_name = MODEL_DISPLAY.get(str(row["tabular_family"]), str(row["tabular_family"]))
        neural_name = MODEL_DISPLAY.get(str(row["neural_family"]), str(row["neural_family"]))
        tabular = f"{tabular_name} {_fmt(row['tabular_mean_test_primary'])}"
        neural = f"{neural_name} {_fmt(row['neural_mean_test_primary'])}"
        strongest_displayed = max(float(row["tabular_mean_test_primary"]), float(row["neural_mean_test_primary"]))
        delta = float(row["mctrcm_mean_primary"]) - strongest_displayed
        rows.append(
            " & ".join(
                [
                    str(row["task_display"]),
                    "$R^2$" if str(row["primary_metric"]) == "r2" else str(row["primary_metric"]).replace("balanced_accuracy", "BA"),
                    _fmt(row["null_mean_test_primary"]),
                    tabular,
                    neural,
                    f"{_fmt(row['mctrcm_mean_primary'])} $\\pm$ {_fmt(row['mctrcm_se_primary'])}",
                    ensemble,
                    _fmt(delta),
                    _fmt(row["mctrcm_rank"], digits=1),
                    str(int(row["validation_selected_k"])),
                ]
            )
            + r" \\"
        )
    if not ranks.empty:
        rank_lookup = {str(row["method"]): float(row["average_rank"]) for _, row in ranks.iterrows()}
        rows.append(r"\midrule")
        rows.append(
            " & ".join(
                [
                    "Average rank",
                    "--",
                    _fmt(rank_lookup.get("null", math.nan), digits=2),
                    _fmt(rank_lookup.get("tabular", math.nan), digits=2),
                    _fmt(rank_lookup.get("neural", math.nan), digits=2),
                    _fmt(rank_lookup.get("mctrcm", math.nan), digits=2),
                    "--",
                    "--",
                    "--",
                    "--",
                ]
            )
            + r" \\"
        )
    _latex_table(
        TABLE_ROOT / "core_main_comparison_clean.tex",
        caption="Core benchmark results. The tabular reference is selected by validation primary metric within each endpoint before test evaluation. It is not the retrospective test-best implemented baseline. Delta compares single-model MC-TRCM with the stronger displayed validation-selected reference; the final row reports average rank.",
        label="tab:core_main_comparison",
        columns="@{}L{0.21\\linewidth}C{0.055\\linewidth}C{0.055\\linewidth}L{0.12\\linewidth}L{0.12\\linewidth}C{0.105\\linewidth}C{0.06\\linewidth}C{0.055\\linewidth}C{0.055\\linewidth}C{0.035\\linewidth}@{}",
        header="Endpoint & Metric & Null & \\makecell[c]{Validation-selected\\\\tabular reference} & \\makecell[c]{Validation-selected\\\\neural reference} & MC-TRCM & Ens. & \\makecell[c]{$\\Delta$ vs stronger\\\\displayed validation\\\\reference} & \\makecell[c]{MC\\\\rank} & $K$ \\\\",
        rows=rows,
    )


def calibration_table() -> None:
    main = pd.read_csv(RESULT_ROOT / "core_main_comparison.csv")
    baseline = pd.read_csv(BASELINE_ROOT / "summary_metrics.csv", keep_default_na=False)
    rows = []
    for task_key in CORE_TASK_ORDER:
        if task_key not in CLASSIFICATION_TASKS:
            continue
        row = main.loc[main["task_key"].eq(task_key)].iloc[0]
        dataset, endpoint = task_key.split("::", 1)
        family = str(row["best_baseline"])
        bframe = baseline.loc[
            baseline["dataset"].eq(dataset)
            & baseline["endpoint"].eq(endpoint)
            & baseline["feature_set"].eq("FULL")
            & baseline["model_family"].eq(family)
        ]
        brier = ece = math.nan
        if not bframe.empty:
            metrics = json.loads(str(bframe.iloc[0]["mean_test_secondary_metrics"]))
            brier = metrics.get("test_brier", math.nan)
            ece = metrics.get("test_ece", math.nan)
        rows.append(
            {
                "task_key": task_key,
                "task_display": TASK_DISPLAY[task_key],
                "baseline_family": family,
                "baseline_brier": brier,
                "baseline_ece": ece,
                "mctrcm_brier": row.get("mctrcm_mean_brier", math.nan),
                "mctrcm_ece": row.get("mctrcm_mean_ece", math.nan),
                "delta_brier": row.get("mctrcm_mean_brier", math.nan) - brier,
                "delta_ece": row.get("mctrcm_mean_ece", math.nan) - ece,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_ROOT / "core_calibration_comparison.csv", index=False)
    latex_rows = []
    for _, row in out.iterrows():
        latex_rows.append(
            " & ".join(
                [
                    row["task_display"],
                    MODEL_DISPLAY.get(str(row["baseline_family"]), str(row["baseline_family"])).replace("_", "\\_"),
                    _fmt(row["baseline_brier"]),
                    _fmt(row["mctrcm_brier"]),
                    _fmt(row["delta_brier"]),
                    _fmt(row["baseline_ece"]),
                    _fmt(row["mctrcm_ece"]),
                    _fmt(row["delta_ece"]),
                ]
            )
            + r" \\"
        )
    _latex_table(
        TABLE_ROOT / "core_calibration_comparison.tex",
        caption="Validation-only calibration comparison for classification endpoints. Negative deltas indicate lower MC-TRCM error.",
        label="tab:core_calibration",
        columns="@{}L{0.31\\linewidth}L{0.11\\linewidth}C{0.075\\linewidth}C{0.075\\linewidth}C{0.06\\linewidth}C{0.075\\linewidth}C{0.075\\linewidth}C{0.06\\linewidth}@{}",
        header="Endpoint & Baseline & \\makecell[c]{Brier\\\\base} & \\makecell[c]{Brier\\\\MC} & $\\Delta$ & \\makecell[c]{ECE\\\\base} & \\makecell[c]{ECE\\\\MC} & $\\Delta$ \\\\",
        rows=latex_rows,
    )


def depth_table() -> None:
    selected = pd.read_csv(RESULT_ROOT / "selected_hyperparameters.csv")
    summary = pd.read_csv(RESULT_ROOT / "k_search_summary.csv")
    rows = []
    for _, sel in selected.iterrows():
        dataset = str(sel["dataset"])
        hp = int(sel["hp_index"])
        subset = summary.loc[summary["dataset"].eq(dataset) & summary["hp_index"].eq(hp)].copy()
        values = {int(row["k"]): row["mean_valid_primary"] for _, row in subset.iterrows()}
        rows.append(
            {
                "dataset": dataset,
                "hp_index": hp,
                "selected_k": int(sel["k"]),
                **{f"k{k}": values.get(k, math.nan) for k in (1, 2, 4, 6, 8)},
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_ROOT / "recursive_depth_selection.csv", index=False)
    latex_rows = []
    for _, row in out.iterrows():
        latex_rows.append(
            " & ".join(
                [
                    DATASET_DISPLAY.get(str(row["dataset"]), str(row["dataset"]).replace("_", "\\_")),
                    str(int(row["hp_index"])),
                    _fmt(row["k1"]),
                    _fmt(row["k2"]),
                    _fmt(row["k4"]),
                    _fmt(row["k6"]),
                    _fmt(row["k8"]),
                    str(int(row["selected_k"])),
                ]
            )
            + r" \\"
        )
    _latex_table(
        TABLE_ROOT / "recursive_depth_selection.tex",
        caption="Validation-selected recursive depth. Values are mean validation primary metric across core tasks for the selected hyperparameter candidate.",
        label="tab:recursive_depth",
        columns="@{}L{0.18\\linewidth}C{0.08\\linewidth}C{0.09\\linewidth}C{0.09\\linewidth}C{0.09\\linewidth}C{0.09\\linewidth}C{0.09\\linewidth}C{0.10\\linewidth}@{}",
        header="Dataset & HP & $K=1$ & $K=2$ & $K=4$ & $K=6$ & $K=8$ & Selected \\\\",
        rows=latex_rows,
    )


def ablation_table() -> None:
    frame = pd.read_csv(RESULT_ROOT / "ablation_summary.csv")
    order = [
        "full",
        "no_film",
        "no_missingness_token",
        "no_missingness_projection",
        "no_pcgrad",
        "no_task_conditioning",
        "no_recursive_refinement",
    ]
    rows = []
    for variant in order:
        sub = frame.loc[frame["variant"].eq(variant)].copy()
        if sub.empty:
            continue
        deprest_delta = sub.loc[sub["dataset"].eq("deprest_cat"), "delta_vs_full"].mean()
        psyche_delta = sub.loc[sub["dataset"].eq("psyche_d"), "delta_vs_full"].mean()
        mean_delta = sub["delta_vs_full"].mean()
        worst_delta = sub["delta_vs_full"].min()
        mean_primary = sub["mean_primary"].mean()
        rows.append(
            {
                "variant": variant,
                "mean_primary": mean_primary,
                "mean_delta": mean_delta,
                "deprest_delta": deprest_delta,
                "psyche_delta": psyche_delta,
                "worst_delta": worst_delta,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_ROOT / "ablation_compact_summary.csv", index=False)
    names = {
        "full": "Full MC-TRCM",
        "no_film": "-FiLM",
        "no_missingness_token": "-missing token",
        "no_missingness_projection": "-missing projection",
        "no_pcgrad": "-PCGrad",
        "no_task_conditioning": "-task conditioning",
        "no_recursive_refinement": "-recursive refinement",
    }
    latex_rows = []
    for _, row in out.iterrows():
        latex_rows.append(
            " & ".join(
                [
                    names.get(row["variant"], row["variant"]).replace("_", "\\_"),
                    _fmt(row["mean_primary"]),
                    _fmt(row["mean_delta"]),
                    _fmt(row["deprest_delta"]),
                    _fmt(row["psyche_delta"]),
                    _fmt(row["worst_delta"]),
                ]
            )
            + r" \\"
        )
    _latex_table(
        TABLE_ROOT / "ablation_compact_summary.tex",
        caption="Five-seed component ablations. Deltas are against Full MC-TRCM and average the endpoint primary metrics within each group.",
        label="tab:ablation_compact",
        columns="@{}L{0.23\\linewidth}C{0.13\\linewidth}C{0.13\\linewidth}C{0.13\\linewidth}C{0.13\\linewidth}C{0.13\\linewidth}@{}",
        header="Variant & \\makecell[c]{Mean\\\\primary} & \\makecell[c]{Mean\\\\$\\Delta$} & \\makecell[c]{DepreST\\\\$\\Delta$} & \\makecell[c]{PSYCHE-D\\\\$\\Delta$} & \\makecell[c]{Worst\\\\$\\Delta$} \\\\",
        rows=latex_rows,
    )


def feature_source_table() -> None:
    frame = pd.read_csv(BASELINE_ROOT / "summary_metrics.csv", keep_default_na=False)
    frame["task_key"] = frame["dataset"].astype(str) + "::" + frame["endpoint"].astype(str)
    frame = frame.loc[frame["task_key"].isin(CORE_TASK_ORDER)].copy()
    for column in ("mean_val_primary", "mean_test_primary"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    mctrcm = _mctrcm_primary_summary().set_index("task_key")
    keep_sets = ["FULL", "SENSOR_ONLY", "SENSOR_VALUES_ONLY", "MISSINGNESS_ONLY", "SYMPTOM_STATIC_CLINICAL"]
    rows = []
    for task_key in CORE_TASK_ORDER:
        task = frame.loc[frame["task_key"].eq(task_key)]
        row = {"task_key": task_key, "task_display": TASK_DISPLAY[task_key]}
        for feature_set in keep_sets:
            subset = task.loc[task["feature_set"].eq(feature_set)]
            if subset.empty:
                row[feature_set] = math.nan
                continue
            idx = subset.groupby("feature_set")["mean_val_primary"].idxmax().iloc[0]
            row[feature_set] = subset.loc[idx, "mean_test_primary"]
        row["MCTRCM_FULL"] = (
            float(mctrcm.loc[task_key, "mean_test_primary"])
            if task_key in mctrcm.index
            else math.nan
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_ROOT / "feature_source_core_summary.csv", index=False)
    latex_rows = []
    for _, row in out.iterrows():
        latex_rows.append(
            " & ".join(
                [
                    row["task_display"],
                    _fmt(row["FULL"]),
                    _fmt(row["MCTRCM_FULL"]),
                    _fmt(row["SENSOR_ONLY"]),
                    _fmt(row["SENSOR_VALUES_ONLY"]),
                    _fmt(row["MISSINGNESS_ONLY"]),
                    _fmt(row["SYMPTOM_STATIC_CLINICAL"]),
                ]
            )
            + r" \\"
        )
    _latex_table(
        TABLE_ROOT / "feature_source_core_summary.tex",
        caption="Feature-source controls and full MC-TRCM comparison. Baseline cells use the validation-selected family within each feature set; the MC-TRCM column reports the full-feature five-seed model.",
        label="tab:feature_source_core",
        columns="@{}L{0.26\\linewidth}C{0.08\\linewidth}C{0.08\\linewidth}C{0.105\\linewidth}C{0.105\\linewidth}C{0.095\\linewidth}C{0.155\\linewidth}@{}",
        header="Endpoint & \\makecell[c]{Full\\\\base} & MC-TRCM & \\makecell[c]{Sensor +\\\\missingness} & \\makecell[c]{Sensor\\\\values} & Missingness & \\makecell[c]{Symptom/static/\\\\clinical} \\\\",
        rows=latex_rows,
    )


def _baseline_seed_summary() -> pd.DataFrame:
    path = ROOT / "results" / "final" / "baseline_results_seeded.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, keep_default_na=False)
    mctrcm = _mctrcm_seed_rows()
    if not mctrcm.empty:
        frame = pd.concat([frame, mctrcm], ignore_index=True, sort=False)
    frame["task_key"] = frame["dataset_id"].astype(str) + "::" + frame["task_name"].astype(str)
    frame["model_name"] = frame["model_name"].astype(str)
    model_order = {model: index for index, model in enumerate(APPENDIX_BASELINE_ORDER)}
    frame["model_order"] = frame["model_name"].map(model_order).fillna(999)
    return frame


def _mctrcm_seed_rows() -> pd.DataFrame:
    path = RESULT_ROOT / "final_seed_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    test = frame.loc[frame["split"].astype(str).eq("test")].copy()
    if test.empty:
        return pd.DataFrame()
    label_types = {
        "deprest_cat::phq9_reg": "continuous",
        "deprest_cat::gad7_reg": "continuous",
        "deprest_cat::phq9_cat": "ordinal",
        "deprest_cat::gad7_cat": "ordinal",
        "psyche_d::phq_change_binary": "binary",
        "psyche_d::phq_change_multiclass": "multiclass",
    }
    rows = []
    for _, row in test.iterrows():
        task_key = str(row["task_key"])
        dataset_id, task_name = task_key.split("::", 1)
        rows.append(
            {
                "experiment_id": str(row["run_name"]),
                "dataset_id": dataset_id,
                "task_name": task_name,
                "model_name": "mctrcm",
                "label_type": label_types.get(task_key, "continuous" if row["primary_metric"] == "r2" else "ordinal"),
                "status": "completed",
                "notes": "core_protocol_final_five_seed",
                "test_auroc": row.get("auroc"),
                "test_auprc": row.get("auprc"),
                "test_balanced_accuracy": row.get("balanced_accuracy"),
                "test_macro_f1": row.get("macro_f1"),
                "test_brier_score": row.get("brier_score"),
                "test_ece": row.get("ece"),
                "test_mae": row.get("mae"),
                "test_rmse": row.get("rmse"),
                "test_r2": row.get("r2"),
                "test_spearman": row.get("spearman"),
                "seed": row.get("seed"),
            }
        )
    return pd.DataFrame(rows)


def _mctrcm_primary_summary() -> pd.DataFrame:
    path = RESULT_ROOT / "final_seed_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame = frame.loc[frame["task_key"].astype(str).isin(CORE_TASK_ORDER)].copy()
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for task_key, group in frame.groupby("task_key", sort=False):
        valid = pd.to_numeric(
            group.loc[group["split"].astype(str).eq("valid"), "primary_value"],
            errors="coerce",
        )
        test = pd.to_numeric(
            group.loc[group["split"].astype(str).eq("test"), "primary_value"],
            errors="coerce",
        )
        rows.append(
            {
                "task_key": task_key,
                "mean_val_primary": float(valid.mean()) if valid.notna().any() else math.nan,
                "mean_test_primary": float(test.mean()) if test.notna().any() else math.nan,
                "se_test_primary": _standard_error(test),
                "n_seeds": int(group.loc[group["split"].astype(str).eq("test"), "seed"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def appendix_baseline_full_regression_table() -> None:
    frame = _baseline_seed_summary()
    if frame.empty:
        return
    reg = frame.loc[frame["label_type"].astype(str).eq("continuous")].copy()
    rows = []
    for (task_key, model_name), group in reg.groupby(["task_key", "model_name"], dropna=False):
        rows.append(
            {
                "task_key": task_key,
                "dataset_id": str(group["dataset_id"].iloc[0]),
                "task_display": APPENDIX_TASK_DISPLAY.get(str(task_key), str(task_key).replace("_", "\\_")),
                "model_name": str(model_name),
                "model_order": float(group["model_order"].iloc[0]),
                "r2_mean": pd.to_numeric(group["test_r2"], errors="coerce").mean(),
                "r2_se": _standard_error(group["test_r2"]),
                "mae_mean": pd.to_numeric(group["test_mae"], errors="coerce").mean(),
                "mae_se": _standard_error(group["test_mae"]),
                "rmse_mean": pd.to_numeric(group["test_rmse"], errors="coerce").mean(),
                "rmse_se": _standard_error(group["test_rmse"]),
                "spearman_mean": pd.to_numeric(group["test_spearman"], errors="coerce").mean(),
                "spearman_se": _standard_error(group["test_spearman"]),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["dataset_id", "task_key", "model_order", "model_name"])
    out.to_csv(RESULT_ROOT / "appendix_baseline_full_regression_metrics.csv", index=False)
    blocks = []
    first = True
    for _, group in out.groupby("task_key", sort=False):
        task_display = str(group["task_display"].iloc[0])
        latex_rows = []
        for _, row in group.iterrows():
            latex_rows.append(
                " & ".join(
                    [
                        MODEL_DISPLAY.get(str(row["model_name"]), str(row["model_name"])).replace("_", "\\_"),
                        _fmt_pm(row["r2_mean"], row["r2_se"]),
                        _fmt_pm(row["mae_mean"], row["mae_se"]),
                        _fmt_pm(row["rmse_mean"], row["rmse_se"]),
                        _fmt_pm(row["spearman_mean"], row["spearman_se"]),
                        str(int(row["n_seeds"])),
                    ]
                )
                + r" \\"
            )
        blocks.append(
            {
                "caption": f"Full-feature regression baseline metrics for {task_display}. Values are test mean and standard error over seeds.",
                "label": f"tab:appendix_full_regression_{_label_suffix(group['task_key'].iloc[0])}",
                "rows": latex_rows,
            }
        )
        first = False
    _latex_table_blocks(
        TABLE_ROOT / "appendix_baseline_full_regression_metrics.tex",
        blocks=blocks,
        columns="@{}p{0.16\\linewidth}p{0.16\\linewidth}p{0.16\\linewidth}p{0.16\\linewidth}p{0.16\\linewidth}p{0.05\\linewidth}@{}",
        header="Model & $R^2$ & MAE & RMSE & Spearman & S \\\\",
        size="\\scriptsize",
        tabcolsep="2pt",
        arraystretch="1.06",
    )


def appendix_baseline_full_classification_table() -> None:
    frame = _baseline_seed_summary()
    if frame.empty:
        return
    cls = frame.loc[~frame["label_type"].astype(str).eq("continuous")].copy()
    rows = []
    for (task_key, model_name), group in cls.groupby(["task_key", "model_name"], dropna=False):
        rows.append(
            {
                "task_key": task_key,
                "dataset_id": str(group["dataset_id"].iloc[0]),
                "task_display": APPENDIX_TASK_DISPLAY.get(str(task_key), str(task_key).replace("_", "\\_")),
                "model_name": str(model_name),
                "model_order": float(group["model_order"].iloc[0]),
                "ba_mean": pd.to_numeric(group["test_balanced_accuracy"], errors="coerce").mean(),
                "ba_se": _standard_error(group["test_balanced_accuracy"]),
                "f1_mean": pd.to_numeric(group["test_macro_f1"], errors="coerce").mean(),
                "f1_se": _standard_error(group["test_macro_f1"]),
                "auroc_mean": pd.to_numeric(group["test_auroc"], errors="coerce").mean(),
                "auroc_se": _standard_error(group["test_auroc"]),
                "auprc_mean": pd.to_numeric(group["test_auprc"], errors="coerce").mean(),
                "auprc_se": _standard_error(group["test_auprc"]),
                "brier_mean": pd.to_numeric(group["test_brier_score"], errors="coerce").mean(),
                "brier_se": _standard_error(group["test_brier_score"]),
                "ece_mean": pd.to_numeric(group["test_ece"], errors="coerce").mean(),
                "ece_se": _standard_error(group["test_ece"]),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["dataset_id", "task_key", "model_order", "model_name"])
    out.to_csv(RESULT_ROOT / "appendix_baseline_full_classification_metrics.csv", index=False)
    blocks = []
    first = True
    for _, group in out.groupby("task_key", sort=False):
        task_display = str(group["task_display"].iloc[0])
        latex_rows = []
        for _, row in group.iterrows():
            latex_rows.append(
                " & ".join(
                    [
                        MODEL_DISPLAY.get(str(row["model_name"]), str(row["model_name"])).replace("_", "\\_"),
                        _fmt_pm(row["ba_mean"], row["ba_se"]),
                        _fmt_pm(row["f1_mean"], row["f1_se"]),
                        _fmt_pm(row["auroc_mean"], row["auroc_se"]),
                        _fmt_pm(row["auprc_mean"], row["auprc_se"]),
                        _fmt_pm(row["brier_mean"], row["brier_se"]),
                        _fmt_pm(row["ece_mean"], row["ece_se"]),
                        str(int(row["n_seeds"])),
                    ]
                )
                + r" \\"
            )
        blocks.append(
            {
                "caption": f"Full-feature classification and ordinal baseline metrics for {task_display}. Values are test mean and standard error over seeds.",
                "label": f"tab:appendix_full_classification_{_label_suffix(group['task_key'].iloc[0])}",
                "rows": latex_rows,
            }
        )
        first = False
    _latex_table_blocks(
        TABLE_ROOT / "appendix_baseline_full_classification_metrics.tex",
        blocks=blocks,
        columns="@{}p{0.12\\linewidth}p{0.105\\linewidth}p{0.105\\linewidth}p{0.105\\linewidth}p{0.105\\linewidth}p{0.105\\linewidth}p{0.105\\linewidth}p{0.04\\linewidth}@{}",
        header="Model & BA & Macro-F1 & AUROC & AUPRC & Brier & ECE & S \\\\",
        size="\\scriptsize",
        tabcolsep="2pt",
        arraystretch="1.06",
    )


def appendix_feature_source_baseline_table() -> None:
    path = BASELINE_ROOT / "summary_metrics.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path, keep_default_na=False)
    frame["model_family"] = frame["model_family"].replace("", "null").astype(str)
    frame = frame.loc[~frame["model_family"].str.contains("mctrcm", case=False, na=False)].copy()
    frame["task_key"] = frame["dataset"].astype(str) + "::" + frame["endpoint"].astype(str)
    model_order = {model: index for index, model in enumerate(APPENDIX_BASELINE_ORDER)}
    frame["model_order"] = frame["model_family"].map(model_order).fillna(999)
    for column in ("mean_val_primary", "mean_test_primary", "se_test_primary"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    out = frame.sort_values(["dataset", "endpoint", "feature_set", "model_order", "model_family"])[
        [
            "task_key",
            "feature_set",
            "model_family",
            "mean_val_primary",
            "mean_test_primary",
            "se_test_primary",
            "n_seeds",
        ]
    ].copy()
    mctrcm = _mctrcm_primary_summary()
    if not mctrcm.empty:
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "task_key": str(row["task_key"]),
                            "feature_set": "FULL",
                            "model_family": "mctrcm",
                            "mean_val_primary": row["mean_val_primary"],
                            "mean_test_primary": row["mean_test_primary"],
                            "se_test_primary": row["se_test_primary"],
                            "n_seeds": row["n_seeds"],
                        }
                        for _, row in mctrcm.iterrows()
                    ]
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    feature_order = {feature_set: index for index, feature_set in enumerate(FEATURE_SET_DISPLAY)}
    model_order = {model: index for index, model in enumerate(APPENDIX_BASELINE_ORDER)}
    out["_feature_order"] = out["feature_set"].map(feature_order).fillna(999)
    out["_model_order"] = out["model_family"].map(model_order).fillna(999)
    out = out.sort_values(["task_key", "_feature_order", "_model_order", "model_family"]).drop(
        columns=["_feature_order", "_model_order"]
    )
    out.to_csv(RESULT_ROOT / "appendix_feature_source_baseline_primary_metrics.csv", index=False)
    blocks = []
    first = True
    for task_key, group in out.groupby("task_key", sort=False):
        task_display = TASK_DISPLAY.get(str(task_key), str(task_key).replace("_", "\\_"))
        latex_rows = []
        for _, row in group.iterrows():
            latex_rows.append(
                " & ".join(
                    [
                        FEATURE_SET_DISPLAY.get(str(row["feature_set"]), str(row["feature_set"]).replace("_", "\\_")),
                        MODEL_DISPLAY.get(str(row["model_family"]), str(row["model_family"])).replace("_", "\\_"),
                        _fmt(row["mean_val_primary"]),
                        _fmt_pm(row["mean_test_primary"], row["se_test_primary"]),
                        str(int(row["n_seeds"])),
                    ]
                )
                + r" \\"
            )
        blocks.append(
            {
                "caption": f"Feature-source baseline primary metrics for {task_display}. Model-family and feature-source choices use the same participant-level split protocol.",
                "label": f"tab:appendix_feature_source_{_label_suffix(task_key)}",
                "rows": latex_rows,
            }
        )
        first = False
    _latex_table_blocks(
        TABLE_ROOT / "appendix_feature_source_baseline_primary_metrics.tex",
        blocks=blocks,
        columns="@{}p{0.20\\linewidth}p{0.13\\linewidth}p{0.10\\linewidth}p{0.15\\linewidth}p{0.04\\linewidth}@{}",
        header="Feature source & Model & Valid & Test & S \\\\",
        size="\\scriptsize",
        tabcolsep="2pt",
        arraystretch="1.04",
    )


def optimization_ablation_table() -> None:
    path = RESULT_ROOT / "ablation_summary.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    order = [
        "full",
        "no_pcgrad",
        "gradnorm",
        "no_uncertainty_weighting",
        "no_task_balanced_sampling",
        "no_conflict_weighting",
    ]
    names = {
        "full": "Full MC-TRCM",
        "no_pcgrad": "-PCGrad",
        "gradnorm": "GradNorm-style balancing",
        "no_uncertainty_weighting": "-uncertainty weighting",
        "no_task_balanced_sampling": "-task-balanced sampling",
        "no_conflict_weighting": "-PCGrad/-uncertainty",
    }
    rows = []
    for variant in order:
        sub = frame.loc[frame["variant"].eq(variant)].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "variant": variant,
                "mean_primary": sub["mean_primary"].mean(),
                "mean_delta": sub["delta_vs_full"].mean(),
                "deprest_delta": sub.loc[sub["dataset"].eq("deprest_cat"), "delta_vs_full"].mean(),
                "psyche_delta": sub.loc[sub["dataset"].eq("psyche_d"), "delta_vs_full"].mean(),
                "n_endpoint_rows": int(len(sub)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_ROOT / "optimization_ablation_summary.csv", index=False)
    latex_rows = []
    for _, row in out.iterrows():
        latex_rows.append(
            " & ".join(
                [
                    names.get(str(row["variant"]), str(row["variant"]).replace("_", "\\_")),
                    _fmt(row["mean_primary"]),
                    _fmt(row["mean_delta"]),
                    _fmt(row["deprest_delta"]),
                    _fmt(row["psyche_delta"]),
                    str(int(row["n_endpoint_rows"])),
                ]
            )
            + r" \\"
        )
    _latex_table(
        TABLE_ROOT / "optimization_ablation_summary.tex",
        caption="Five-seed optimization ablations. Deltas are against the full validation-selected MC-TRCM configuration.",
        label="tab:optimization_ablation",
        columns="@{}L{0.25\\linewidth}C{0.14\\linewidth}C{0.14\\linewidth}C{0.14\\linewidth}C{0.14\\linewidth}C{0.08\\linewidth}@{}",
        header="Variant & \\makecell[c]{Mean\\\\primary} & \\makecell[c]{Mean\\\\$\\Delta$} & \\makecell[c]{DepreST\\\\$\\Delta$} & \\makecell[c]{PSYCHE-D\\\\$\\Delta$} & Rows \\\\",
        rows=latex_rows,
    )


def success_summary() -> None:
    main = pd.read_csv(RESULT_ROOT / "core_main_comparison.csv")
    rows = []
    best_or_tie = 0
    severe_degradation = 0
    for _, row in main.iterrows():
        m = float(row["mctrcm_mean_primary"])
        tab = float(row["tabular_mean_test_primary"])
        neu = float(row["neural_mean_test_primary"])
        if tab >= neu:
            b = tab
            se = float(row["tabular_se_test_primary"]) if pd.notna(row["tabular_se_test_primary"]) else 0.0
        else:
            b = neu
            se = float(row["neural_se_test_primary"]) if pd.notna(row["neural_se_test_primary"]) else 0.0
        delta = m - b
        tied = abs(delta) <= max(0.01, 2.0 * se)
        best = delta > 0
        if best or tied:
            best_or_tie += 1
        if delta < -0.05:
            severe_degradation += 1
        rows.append(
            {
                "task_key": row["task_key"],
                "task_display": row["task_display"],
                "delta_vs_best_baseline": delta,
                "best_or_tied_by_rule": bool(best or tied),
            }
        )
    ranks = pd.read_csv(RESULT_ROOT / "core_average_ranks.csv")
    tab_rank = float(ranks.loc[ranks["method"].eq("tabular"), "average_rank"].iloc[0])
    mc_rank = float(ranks.loc[ranks["method"].eq("mctrcm"), "average_rank"].iloc[0])
    summary = pd.DataFrame(rows)
    summary["best_or_tied_count"] = best_or_tie
    summary["mctrcm_average_rank"] = mc_rank
    summary["tabular_average_rank"] = tab_rank
    summary["primary_success"] = (best_or_tie >= 4) and (mc_rank <= tab_rank) and (severe_degradation == 0)
    summary.to_csv(RESULT_ROOT / "success_criteria_summary.csv", index=False)


def main() -> None:
    main_comparison()
    calibration_table()
    depth_table()
    ablation_table()
    feature_source_table()
    appendix_baseline_full_regression_table()
    appendix_baseline_full_classification_table()
    appendix_feature_source_baseline_table()
    optimization_ablation_table()
    success_summary()


if __name__ == "__main__":
    main()
