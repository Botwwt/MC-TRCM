from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    family: str
    implemented: bool
    dependency: str | None
    note: str


BASELINE_SPECS = {
    "elastic_net": BaselineSpec(
        name="elastic_net",
        family="tabular",
        implemented=True,
        dependency="sklearn",
        note="Elastic Net regression and elastic-net logistic regression.",
    ),
    "lightgbm": BaselineSpec(
        name="lightgbm",
        family="tabular",
        implemented=True,
        dependency="lightgbm",
        note="Gradient boosting tree baseline with dataset-consistent features.",
    ),
    "xgboost": BaselineSpec(
        name="xgboost",
        family="tabular",
        implemented=True,
        dependency="xgboost",
        note="Histogram tree boosting baseline with matched split protocol.",
    ),
    "ebm": BaselineSpec(
        name="ebm",
        family="tabular",
        implemented=True,
        dependency="interpret",
        note="Explainable Boosting Machine baseline for intrinsic interpretability.",
    ),
    "mlp_small": BaselineSpec(
        name="mlp_small",
        family="tabular",
        implemented=True,
        dependency="sklearn",
        note="Parameter-constrained one-hidden-layer MLP baseline.",
    ),
    "mlp": BaselineSpec(
        name="mlp",
        family="torch_tabular",
        implemented=True,
        dependency="torch",
        note="Capacity-aligned tabular MLP matched to the current best MC-TRCM-v2 parameter budget.",
    ),
    "gru_small": BaselineSpec(
        name="gru_small",
        family="sequence",
        implemented=True,
        dependency="torch",
        note="Tiny GRU over a four-token multi-timescale window sequence view.",
    ),
    "gru": BaselineSpec(
        name="gru",
        family="capacity_sequence",
        implemented=True,
        dependency="torch",
        note="Capacity-aligned GRU over a four-token multi-timescale window sequence view.",
    ),
    "lstm_small": BaselineSpec(
        name="lstm_small",
        family="sequence",
        implemented=True,
        dependency="torch",
        note="Tiny LSTM over a four-token multi-timescale window sequence view.",
    ),
    "lstm": BaselineSpec(
        name="lstm",
        family="capacity_sequence",
        implemented=True,
        dependency="torch",
        note="Capacity-aligned LSTM over a four-token multi-timescale window sequence view.",
    ),
    "transformer_small": BaselineSpec(
        name="transformer_small",
        family="sequence",
        implemented=True,
        dependency="torch",
        note="Matched-size Transformer over a four-token multi-timescale window sequence view.",
    ),
    "transformer": BaselineSpec(
        name="transformer",
        family="capacity_sequence",
        implemented=True,
        dependency="torch",
        note="Capacity-aligned Transformer over a four-token multi-timescale window sequence view.",
    ),
    "psyche_d_public_two_stage": BaselineSpec(
        name="psyche_d_public_two_stage",
        family="dataset_specific",
        implemented=False,
        dependency="lightgbm",
        note="Implemented as a dedicated sidecar audit via src/models/run_psyche_d_public_baseline.py; not part of the unified participant-level baseline runner.",
    ),
    "deprest_cat_public_time_series": BaselineSpec(
        name="deprest_cat_public_time_series",
        family="dataset_specific",
        implemented=False,
        dependency=None,
        note="Implemented as a dedicated sidecar audit via src/models/run_deprest_cat_public_baseline.py; not part of the unified participant-level baseline runner.",
    ),
    "depresjon_public_actigraphy": BaselineSpec(
        name="depresjon_public_actigraphy",
        family="dataset_specific",
        implemented=False,
        dependency=None,
        note="Implemented as a dedicated sidecar audit via src/models/run_depresjon_public_baseline.py; the public Depresjon_ML repository is preserved for transparency but excluded from the main benchmark table because it uses target-derived inputs and random sequence-level splitting.",
    ),
}

FIRST_WAVE_MODELS = ("elastic_net", "lightgbm", "xgboost", "ebm", "mlp", "gru", "lstm", "transformer")


def probe_dependencies() -> dict[str, bool]:
    return {
        "sklearn": bool(importlib.util.find_spec("sklearn")),
        "lightgbm": bool(importlib.util.find_spec("lightgbm")),
        "xgboost": bool(importlib.util.find_spec("xgboost")),
        "interpret": bool(importlib.util.find_spec("interpret")),
        "torch": bool(importlib.util.find_spec("torch")),
    }


def get_baseline_spec(model_name: str) -> BaselineSpec:
    if model_name not in BASELINE_SPECS:
        raise KeyError(f"Unknown baseline model: {model_name}")
    return BASELINE_SPECS[model_name]


def build_estimator(
    model_name: str,
    label_type: str,
    seed: int,
    n_classes: int,
):
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    is_regression = label_type == "continuous"

    if model_name == "elastic_net":
        if is_regression:
            from sklearn.linear_model import ElasticNet

            model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=seed)
        else:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(
                solver="saga",
                l1_ratio=0.5,
                max_iter=4000,
                class_weight="balanced",
                random_state=seed,
            )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    if model_name == "mlp_small":
        if is_regression:
            model = MLPRegressor(
                hidden_layer_sizes=(32,),
                activation="relu",
                alpha=1e-4,
                batch_size="auto",
                learning_rate_init=1e-3,
                max_iter=400,
                early_stopping=True,
                random_state=seed,
            )
        else:
            model = MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="relu",
                alpha=1e-4,
                batch_size="auto",
                learning_rate_init=1e-3,
                max_iter=400,
                early_stopping=True,
                random_state=seed,
            )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    if model_name == "lightgbm":
        if is_regression:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=10,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                verbosity=-1,
            )
        else:
            from lightgbm import LGBMClassifier

            objective = "multiclass" if n_classes > 2 else "binary"
            model = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=10,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                objective=objective,
                random_state=seed,
                verbosity=-1,
            )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )

    if model_name == "xgboost":
        if is_regression:
            from xgboost import XGBRegressor

            model = XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=seed,
                tree_method="hist",
            )
        else:
            from xgboost import XGBClassifier

            if n_classes > 2:
                model = XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=seed,
                    tree_method="hist",
                    objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss",
                )
            else:
                model = XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=seed,
                    tree_method="hist",
                    objective="binary:logistic",
                    eval_metric="logloss",
                )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )

    if model_name == "ebm":
        if is_regression:
            from interpret.glassbox import ExplainableBoostingRegressor

            model = ExplainableBoostingRegressor(
                max_bins=64,
                interactions=0,
                outer_bags=8,
                n_jobs=1,
                random_state=seed,
            )
        else:
            from interpret.glassbox import ExplainableBoostingClassifier

            model = ExplainableBoostingClassifier(
                max_bins=64,
                interactions=0,
                outer_bags=8,
                n_jobs=1,
                random_state=seed,
            )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )

    raise NotImplementedError(f"Estimator builder missing for model={model_name}")
