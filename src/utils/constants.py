from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_IDS = (
    "studentlife",
    "deprest_cat",
    "psyche_d",
    "depresjon",
    "obf",
)

CANONICAL_MODALITIES = (
    "activity",
    "sleep",
    "communication",
    "phone_use",
    "mobility",
    "static",
    "symptom_context",
)

WINDOW_SPECS = {
    "short": (1, 3),
    "medium": (7, 7),
    "long": (14, 30),
}

FEATURE_STATISTICS = (
    "mean",
    "sd",
    "cv",
    "slope",
    "entropy",
    "day_night_ratio",
    "weekend_shift",
    "regularity",
    "interevent_cv",
    "ratio",
    "missing_ratio",
)

FEATURE_FAMILIES = (
    "activity",
    "sleep",
    "communication",
    "phone_use",
    "mobility",
    "symptom_context",
)

CONCEPT_DEFINITIONS = {
    "c1": "Psychomotor activity level",
    "c2": "Activity irregularity",
    "c3": "Sleep disruption",
    "c4": "Circadian fragmentation",
    "c5": "Mobility breadth",
    "c6": "Social communication engagement",
    "c7": "Communication volatility",
    "c8": "Recent symptom / vulnerability context",
}


def concept_keys_for_dim(concept_dim: int) -> list[str]:
    concept_dim = max(int(concept_dim), 0)
    base_keys = list(CONCEPT_DEFINITIONS)
    keys = base_keys[:concept_dim]
    for concept_index in range(len(keys), concept_dim):
        keys.append(f"aux_c{concept_index + 1}")
    return keys

DATASET_CONCEPT_AVAILABILITY = {
    "studentlife": {
        "c1": 1,
        "c2": 1,
        "c3": 1,
        "c4": 1,
        "c5": 1,
        "c6": 1,
        "c7": 1,
        "c8": 1,
    },
    "deprest_cat": {
        "c1": 0,
        "c2": 0,
        "c3": 0,
        "c4": 0,
        "c5": 0,
        "c6": 1,
        "c7": 1,
        "c8": 1,
    },
    "psyche_d": {
        "c1": 1,
        "c2": 1,
        "c3": 1,
        "c4": 1,
        "c5": 0,
        "c6": 0,
        "c7": 0,
        "c8": 1,
    },
    "depresjon": {
        "c1": 1,
        "c2": 1,
        "c3": 1,
        "c4": 1,
        "c5": 0,
        "c6": 0,
        "c7": 0,
        "c8": 0,
    },
    "obf": {
        "c1": 1,
        "c2": 1,
        "c3": 1,
        "c4": 1,
        "c5": 0,
        "c6": 0,
        "c7": 0,
        "c8": 0,
    },
}
