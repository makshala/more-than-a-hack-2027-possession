#!/usr/bin/env python3
"""Train Role 4 H/A possession models on the external SkillCorner V3 dataset.

This script intentionally never reads the target match, target video, or target
match ground truth. It only consumes the supplied external-training Parquet and
its feature manifest.

Expected files (by default):
  skillcorner_training_v3.parquet
  skillcorner_feature_manifest_v3.json

The manifest must identify the exact 37 model-input columns. The script does
not guess feature names and does not silently train on every numeric column.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

LOG = logging.getLogger("role4_training")

DEFAULT_PARQUET = "skillcorner_training_v3.parquet"
DEFAULT_MANIFEST = "skillcorner_feature_manifest_v3.json"
EXPECTED_FEATURE_COUNT = 37

# These are column-name conventions only; the selected names still have to be
# present in the supplied Parquet. An explicit --frame-column overrides them.
FRAME_INDEX_CANDIDATES = (
    "frame_idx",
    "frame_index",
    "frame_number",
    "frame_num",
    "frame_id",
    "frame",
)
TIME_SECONDS_CANDIDATES = (
    "timestamp_seconds",
    "timestamp_s",
    "time_seconds",
    "time_s",
    "timestamp",
    "game_time_seconds",
    "elapsed_seconds",
)
TIME_MILLISECONDS_CANDIDATES = ("timestamp_ms", "time_ms", "elapsed_ms")

# Keys commonly used by feature manifests. We require an exact list of 37.
FEATURE_LIST_KEYS = (
    "model_input_features",
    "model_features",
    "model_feature_columns",
    "model_input_columns",
    "model_features_v2",
    "feature_columns",
    "selected_features",
    "features",
)
FEATURE_NAME_KEYS = ("name", "column", "column_name", "feature", "feature_name")


def _normalise_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _names_from_feature_list(value: Any) -> list[str] | None:
    """Convert a manifest feature list into column names, if possible."""
    if not isinstance(value, list) or not value:
        return None

    if all(isinstance(item, str) and item.strip() for item in value):
        return [item.strip() for item in value]

    if all(isinstance(item, dict) for item in value):
        names: list[str] = []
        for item in value:
            # If the manifest explicitly flags model inputs, respect that flag.
            role = str(item.get("role", item.get("usage", ""))).lower()
            include = item.get("model_input", item.get("use_for_model", None))
            is_model_feature = (
                include is True
                or role in {"model", "model_input", "input", "feature"}
            )
            name = next(
                (
                    item.get(key)
                    for key in FEATURE_NAME_KEYS
                    if isinstance(item.get(key), str) and item.get(key).strip()
                ),
                None,
            )
            if name is None:
                return None
            # If any item has a model-input marker, only retain marked entries.
            names.append((name.strip(), is_model_feature, include, role))

        has_markers = any(flag or inc is not None or role for _, flag, inc, role in names)
        if has_markers:
            filtered = [name for name, flag, inc, role in names if flag]
            return filtered or None
        return [name for name, _, _, _ in names]

    return None


def load_feature_names(
    manifest_path: Path | None,
    explicit_features: str | None,
    expected_count: int,
) -> tuple[list[str], str]:
    """Load exactly the model feature list, preferably from the V3 manifest.

    --feature-columns can be used as an explicit override (comma-separated).
    Otherwise, common manifest structures are searched recursively. A candidate
    must contain exactly `expected_count` names; no feature list is invented.
    """
    if explicit_features:
        names = [part.strip() for part in explicit_features.split(",") if part.strip()]
        source = "--feature-columns"
    else:
        if manifest_path is None or not manifest_path.is_file():
            raise FileNotFoundError(
                "Nie znaleziono manifestu cech. Podaj --manifest albo jawne "
                "--feature-columns z dokładnymi nazwami 37 cech z V3."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        key_priority = {_normalise_key(key): i for i, key in enumerate(FEATURE_LIST_KEYS)}
        candidates: list[tuple[int, int, str, list[str]]] = []

        def walk(node: Any, path: str = "$") -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    norm_key = _normalise_key(str(key))
                    if norm_key in key_priority:
                        candidate = _names_from_feature_list(value)
                        if candidate:
                            candidates.append(
                                (0 if len(candidate) == expected_count else 1,
                                 key_priority[norm_key], path + "." + str(key), candidate)
                            )
                    walk(value, path + "." + str(key))
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")

        walk(manifest)
        exact = [item for item in candidates if len(item[3]) == expected_count]
        if not exact:
            discovered = sorted({f"{path}: {len(names)}" for _, _, path, names in candidates})
            details = "; ".join(discovered) if discovered else "brak rozpoznanej listy"
            raise ValueError(
                f"Manifest nie wskazuje listy dokładnie {expected_count} cech modelowych. "
                f"Rozpoznane kandydatury: {details}. Podaj --feature-columns, "
                "jeżeli schemat manifestu ma inną strukturę."
            )
        exact.sort(key=lambda item: (item[0], item[1], item[2]))
        _, _, source, names = exact[0]

    if len(names) != expected_count:
        raise ValueError(
            f"Kontrakt wymaga {expected_count} cech modelowych, otrzymano {len(names)} "
            f"z {source}."
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Lista cech zawiera duplikaty: {duplicates}")
    return names, source


def validate_and_prepare_data(
    df: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
    match_column: str,
    max_nan_ratio: float,
    frame_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any], pd.DataFrame, str | None, str | None]:
    """Strictly validate the V3 table and return only the declared model inputs.

    NaNs are measured, not globally rejected: they are expected in some
    observability/proximity features and are handled by the LR imputer or the
    native missing-value support of HistGradientBoostingClassifier.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dane wejściowe nie są obiektem pandas.DataFrame.")
    if df.empty:
        raise ValueError("Parquet jest pusty — brak próbek treningowych.")
    if df.columns.duplicated().any():
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Parquet zawiera zduplikowane nazwy kolumn: {duplicate_columns}")

    required_aux = [target_column, match_column]
    missing = [name for name in feature_names + required_aux if name not in df.columns]
    if missing:
        raise ValueError(f"Naruszenie Data Contract — brak wymaganych kolumn: {missing}")
    if target_column in feature_names or match_column in feature_names:
        raise ValueError("target i match_id nie mogą należeć do 37 cech modelowych.")

    # Target must already be a valid binary label. Do not silently drop or repair.
    raw_target = df[target_column]
    if raw_target.isna().any():
        count = int(raw_target.isna().sum())
        raise ValueError(f"Target '{target_column}' zawiera {count} wartości NaN.")
    try:
        target_numeric = pd.to_numeric(raw_target, errors="raise")
    except Exception as exc:
        raise ValueError(f"Target '{target_column}' nie jest numeryczny 0/1.") from exc
    target_values = set(pd.unique(target_numeric).tolist())
    if not target_values.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"Target '{target_column}' zawiera wartości spoza {{0, 1}}: "
            f"{sorted(map(str, target_values))[:20]}"
        )
    y = target_numeric.astype("int8").reset_index(drop=True)
    class_counts = y.value_counts().reindex([0, 1], fill_value=0)
    if (class_counts == 0).any():
        raise ValueError(
            "Target musi zawierać obie klasy H=0 i A=1. "
            f"Liczności: H={int(class_counts.loc[0])}, A={int(class_counts.loc[1])}."
        )

    groups = df[match_column]
    if groups.isna().any():
        raise ValueError(f"Kolumna '{match_column}' zawiera brakujące identyfikatory meczów.")
    groups = groups.reset_index(drop=True)
    if groups.nunique() < 2:
        raise ValueError("Walidacja GroupKFold wymaga co najmniej dwóch różnych match_id.")

    # Only the 37 manifest-declared columns are sent to the estimators.
    X = df.loc[:, feature_names].copy().reset_index(drop=True)
    bad_dtypes: list[str] = []
    for column in feature_names:
        if not pd.api.types.is_numeric_dtype(X[column].dtype):
            try:
                X[column] = pd.to_numeric(X[column], errors="raise")
            except Exception:
                bad_dtypes.append(f"{column} ({X[column].dtype})")
    if bad_dtypes:
        raise TypeError(
            "Wszystkie 37 cech muszą być numeryczne. Nie można przekonwertować: "
            + ", ".join(bad_dtypes)
        )

    # Convert +/-inf into a contract error instead of letting estimators fail later.
    numeric_values = X.to_numpy(dtype=np.float64, copy=False, na_value=np.nan)
    inf_mask = np.isinf(numeric_values)
    if inf_mask.any():
        locations = np.argwhere(inf_mask)
        affected = sorted({feature_names[int(col)] for _, col in locations[:100]})
        raise ValueError(
            f"Cechy zawierają {int(inf_mask.sum())} wartości +/-inf. "
            f"Kolumny (maks. 100): {affected}"
        )

    nan_by_feature = X.isna().mean().rename("nan_ratio").reset_index()
    nan_by_feature.columns = ["feature", "nan_ratio"]
    nan_by_feature["nan_count"] = X.isna().sum().values.astype(int)
    nan_by_feature["row_count"] = len(X)
    overall_nan_ratio = float(X.isna().to_numpy().mean())
    excessive_nan = nan_by_feature.loc[nan_by_feature["nan_ratio"] > max_nan_ratio]
    if not excessive_nan.empty:
        bad = excessive_nan[["feature", "nan_ratio"]].to_dict("records")
        raise ValueError(
            f"Naruszenie Data Contract — kolumny przekraczają --max-nan-ratio="
            f"{max_nan_ratio:.3f}: {bad}"
        )

    all_nan_features = nan_by_feature.loc[nan_by_feature["nan_ratio"] == 1.0, "feature"].tolist()
    if all_nan_features:
        warnings.warn(
            "Następujące cechy są w całości NaN w danych treningowych: "
            f"{all_nan_features}. Zachowam je w pipeline LR jako cechy puste, "
            "ale nie będą niosły sygnału uczonego na tych danych.",
            RuntimeWarning,
        )

    # Optional temporal ordering key. No model input is added by this step.
    sequence_column, sequence_kind = resolve_sequence_column(df, frame_column)
    if sequence_column is not None:
        seq = pd.to_numeric(df[sequence_column], errors="coerce")
        if seq.isna().any() or not np.isfinite(seq.to_numpy(dtype=float)).all():
            warnings.warn(
                f"Kolumna sekwencji '{sequence_column}' zawiera wartości nienumeryczne/NaN; "
                "metryki przejść policzę po kolejności wierszy w Parquet.",
                RuntimeWarning,
            )
            sequence_column, sequence_kind = None, None
        else:
            key_frame = pd.DataFrame({"match": groups, "seq": seq.reset_index(drop=True)})
            duplicates = key_frame.duplicated(["match", "seq"], keep=False)
            if duplicates.any():
                sample = key_frame.loc[duplicates].head(10).to_dict("records")
                raise ValueError(
                    f"Duplikaty ({match_column}, {sequence_column}) — kontrakt jednego "
                    f"wiersza na klatkę został naruszony. Przykłady: {sample}"
                )

    class_total = int(len(y))
    h_count, a_count = int(class_counts.loc[0]), int(class_counts.loc[1])
    dataset_summary: dict[str, Any] = {
        "rows": class_total,
        "matches": int(groups.nunique()),
        "feature_count": len(feature_names),
        "target_column": target_column,
        "match_column": match_column,
        "class_counts": {"H_0": h_count, "A_1": a_count},
        "class_shares": {"H_0": h_count / class_total, "A_1": a_count / class_total},
        "class_imbalance_ratio_majority_to_minority": max(h_count, a_count) / min(h_count, a_count),
        "nan_ratio_overall": overall_nan_ratio,
        "nan_cells": int(X.isna().to_numpy().sum()),
        "sequence_column": sequence_column,
        "sequence_kind": sequence_kind,
        "all_nan_features": all_nan_features,
    }

    if dataset_summary["class_imbalance_ratio_majority_to_minority"] >= 10:
        warnings.warn(
            "Silny imbalance klas (co najmniej 10:1). LR użyje class_weight='balanced', "
            "a HGB dostanie wagi próbek wyliczone wyłącznie z etykiet treningowego folda.",
            RuntimeWarning,
        )

    return X, y, groups, dataset_summary, nan_by_feature, sequence_column, sequence_kind


def resolve_sequence_column(
    df: pd.DataFrame, requested: str | None
) -> tuple[str | None, str | None]:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Wskazana kolumna sekwencji '{requested}' nie istnieje.")
        return requested, infer_sequence_kind(requested)

    lower_to_original = {str(column).lower(): str(column) for column in df.columns}
    for candidate in FRAME_INDEX_CANDIDATES:
        if candidate in lower_to_original:
            column = lower_to_original[candidate]
            return column, "frame_index"
    for candidate in TIME_SECONDS_CANDIDATES:
        if candidate in lower_to_original:
            column = lower_to_original[candidate]
            return column, "time_seconds"
    for candidate in TIME_MILLISECONDS_CANDIDATES:
        if candidate in lower_to_original:
            column = lower_to_original[candidate]
            return column, "time_milliseconds"
    return None, None


def infer_sequence_kind(column: str) -> str:
    lower = column.lower()
    if lower in TIME_MILLISECONDS_CANDIDATES or lower.endswith("_ms"):
        return "time_milliseconds"
    if lower in TIME_SECONDS_CANDIDATES or "timestamp" in lower or lower.endswith("_seconds"):
        return "time_seconds"
    return "frame_index"


def make_estimators(random_state: int) -> dict[str, Any]:
    imputer_kwargs: dict[str, Any] = {"strategy": "median"}
    if "keep_empty_features" in inspect.signature(SimpleImputer).parameters:
        imputer_kwargs["keep_empty_features"] = True

    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(**imputer_kwargs)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    # HistGradientBoostingClassifier handles NaNs natively. No imputer/scaler is
    # needed, and early_stopping is disabled to avoid an internal row-wise split.
    hist_gradient_boosting = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=random_state,
    )
    return {
        "logistic_regression": logistic,
        "hist_gradient_boosting": hist_gradient_boosting,
    }


def _positive_class_probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(estimator.classes_)
    positive_idx = np.flatnonzero(classes == 1)
    if len(positive_idx) != 1:
        raise RuntimeError(f"Model nie ma jednoznacznej klasy A=1: classes_={classes}")
    return estimator.predict_proba(X)[:, int(positive_idx[0])]


def fit_estimator(estimator: Any, model_name: str, X: pd.DataFrame, y: pd.Series) -> Any:
    if model_name == "hist_gradient_boosting":
        weights = compute_sample_weight(class_weight="balanced", y=y)
        estimator.fit(X, y, sample_weight=weights)
    else:
        estimator.fit(X, y)
    return estimator


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f_weighted),
        "H_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "A_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }


def build_match_sequences(
    groups: pd.Series,
    df: pd.DataFrame,
    sequence_column: str | None,
    sequence_kind: str | None,
    fps: float,
) -> list[np.ndarray]:
    """Return positional row indices, ordered and split at observed time gaps."""
    sequence: pd.Series | None = None
    if sequence_column is not None:
        sequence = pd.to_numeric(df[sequence_column], errors="coerce").reset_index(drop=True)

    sequences: list[np.ndarray] = []
    group_values = groups.to_numpy()
    for match_id in pd.unique(group_values):
        positions = np.flatnonzero(group_values == match_id)
        if sequence is not None:
            order = np.argsort(sequence.iloc[positions].to_numpy(dtype=float), kind="stable")
            positions = positions[order]
        if len(positions) == 0:
            continue

        if sequence is None or len(positions) == 1:
            sequences.append(positions)
            continue

        values = sequence.iloc[positions].to_numpy(dtype=float)
        if sequence_kind == "time_seconds":
            max_gap = 1.5 / fps
        elif sequence_kind == "time_milliseconds":
            max_gap = 1500.0 / fps
        else:
            max_gap = 1.5
        split_points = np.flatnonzero(np.diff(values) > max_gap) + 1
        for block in np.split(positions, split_points):
            if len(block):
                sequences.append(block)
    return sequences


def transition_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sequences: list[np.ndarray],
) -> dict[str, Any]:
    true_transitions = 0
    predicted_transitions = 0
    adjacent_pairs = 0
    for positions in sequences:
        if len(positions) < 2:
            continue
        yt = y_true[positions]
        yp = y_pred[positions]
        adjacent_pairs += len(positions) - 1
        true_transitions += int(np.count_nonzero(yt[1:] != yt[:-1]))
        predicted_transitions += int(np.count_nonzero(yp[1:] != yp[:-1]))

    ratio = (
        float(predicted_transitions / true_transitions)
        if true_transitions > 0
        else None
    )
    return {
        "true_transition_count": int(true_transitions),
        "predicted_transition_count": int(predicted_transitions),
        "transition_count_ratio_predicted_over_true": ratio,
        "adjacent_observed_frame_pairs": int(adjacent_pairs),
        "predicted_transitions_per_1000_adjacent_pairs": (
            float(1000.0 * predicted_transitions / adjacent_pairs)
            if adjacent_pairs > 0
            else None
        ),
        "interpretation": (
            "Wartość >1 oznacza więcej przełączeń H/A niż w etykietach; "
            "wartość <1 oznacza mniej przełączeń. To proxy migotania, nie oficjalny spellF1."
        ),
    }


def evaluate_oof(
    y: np.ndarray,
    predictions: np.ndarray,
    groups: pd.Series,
    sequences: list[np.ndarray],
) -> dict[str, Any]:
    overall = classification_metrics(y, predictions)
    temporal = transition_statistics(y, predictions, sequences)

    per_match: dict[str, Any] = {}
    match_macro_f1: list[float] = []
    for match_id in pd.unique(groups.to_numpy()):
        positions = np.flatnonzero(groups.to_numpy() == match_id)
        metrics = classification_metrics(y[positions], predictions[positions])
        per_match[str(match_id)] = metrics
        match_macro_f1.append(metrics["f1_macro"])

    return {
        "frame_level_oof": overall,
        "equal_weight_mean_match_f1_macro": (
            float(np.mean(match_macro_f1)) if match_macro_f1 else None
        ),
        "temporal_stability_proxy": temporal,
        "per_match_oof": per_match,
    }


def run_grouped_oof(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
    random_state: int,
    df_for_sequence: pd.DataFrame,
    sequence_column: str | None,
    sequence_kind: str | None,
    fps: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    unique_matches = int(groups.nunique())
    if n_splits < 2 or n_splits > unique_matches:
        raise ValueError(
            f"GroupKFold n_splits={n_splits} jest nieprawidłowe dla {unique_matches} meczów."
        )

    splitter = GroupKFold(n_splits=n_splits)
    y_array = y.to_numpy(dtype=np.int8)
    model_names = ("logistic_regression", "hist_gradient_boosting")
    oof_probs = {name: np.full(len(X), np.nan, dtype=float) for name in model_names}
    oof_preds = {name: np.full(len(X), -1, dtype=np.int8) for name in model_names}
    fold_reports: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        splitter.split(X, y_array, groups=groups), start=1
    ):
        train_matches = set(groups.iloc[train_idx].tolist())
        valid_matches = set(groups.iloc[valid_idx].tolist())
        overlap = train_matches.intersection(valid_matches)
        if overlap:
            raise RuntimeError(f"Leakage match_id pomiędzy train i validation: {overlap}")
        if len(np.unique(y_array[train_idx])) != 2:
            raise ValueError(
                f"Fold {fold_number}: część treningowa nie zawiera obu klas H/A. "
                "Nie można bezpiecznie wytrenować klasyfikatora binarnego."
            )

        fold_report: dict[str, Any] = {
            "fold": fold_number,
            "train_matches": sorted(map(str, train_matches)),
            "validation_matches": sorted(map(str, valid_matches)),
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
            "train_class_counts": {
                "H_0": int(np.sum(y_array[train_idx] == 0)),
                "A_1": int(np.sum(y_array[train_idx] == 1)),
            },
        }

        for model_name, estimator in make_estimators(random_state + fold_number).items():
            fit_estimator(estimator, model_name, X.iloc[train_idx], y.iloc[train_idx])
            probabilities = _positive_class_probability(estimator, X.iloc[valid_idx])
            predictions = (probabilities >= 0.5).astype(np.int8)
            oof_probs[model_name][valid_idx] = probabilities
            oof_preds[model_name][valid_idx] = predictions
            fold_report.setdefault("models", {})[model_name] = classification_metrics(
                y_array[valid_idx], predictions
            )
        fold_reports.append(fold_report)

    evaluation_sequences = build_match_sequences(
        groups, df_for_sequence, sequence_column, sequence_kind, fps
    )
    model_reports: dict[str, Any] = {}
    for model_name in model_names:
        if np.isnan(oof_probs[model_name]).any() or np.any(oof_preds[model_name] < 0):
            raise RuntimeError(f"OOF predykcje nie pokrywają wszystkich próbek dla {model_name}.")
        model_reports[model_name] = evaluate_oof(
            y_array, oof_preds[model_name], groups, evaluation_sequences
        )

    cv_summary = {
        "strategy": "GroupKFold(match_id)",
        "n_splits": n_splits,
        "folds": fold_reports,
        "oof_rows": int(len(X)),
        "random_frame_split_used": False,
        "threshold_for_frame_prediction": 0.5,
    }
    return cv_summary, oof_probs, oof_preds, fold_reports, model_reports


def train_final_estimators(
    X: pd.DataFrame, y: pd.Series, random_state: int
) -> dict[str, Any]:
    """Fit deployable artifacts on the complete external SkillCorner dataset."""
    final_estimators: dict[str, Any] = {}
    for model_name, estimator in make_estimators(random_state).items():
        final_estimators[model_name] = fit_estimator(estimator, model_name, X, y)
    return final_estimators


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Role 4 H/A models on external SkillCorner V3 data only."
    )
    parser.add_argument("--parquet", default=DEFAULT_PARQUET, help="Ścieżka do Parquet V3.")
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Ścieżka do skillcorner_feature_manifest_v3.json. Domyślnie szukany "
            "obok Parquet."
        ),
    )
    parser.add_argument(
        "--feature-columns",
        default=None,
        help="Jawna lista 37 nazw kolumn, rozdzielona przecinkami (override manifestu).",
    )
    parser.add_argument("--expected-feature-count", type=int, default=EXPECTED_FEATURE_COUNT)
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--match-column", default="match_id")
    parser.add_argument(
        "--frame-column",
        default=None,
        help="Opcjonalna kolumna kolejności klatek/czasu używana tylko w metrykach temporalnych.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="FPS danych SkillCorner (domyślnie 10).")
    parser.add_argument("--n-splits", type=int, default=5, help="Liczba foldów GroupKFold (domyślnie 5).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-nan-ratio",
        type=float,
        default=1.0,
        help="Maksymalny dopuszczalny udział NaN w pojedynczej cesze; 1.0 = raportuj, nie odrzucaj.",
    )
    parser.add_argument("--output-dir", default="role4_artifacts")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.fps <= 0:
        raise ValueError("--fps musi być większe od zera.")
    if not 0.0 <= args.max_nan_ratio <= 1.0:
        raise ValueError("--max-nan-ratio musi mieścić się w przedziale [0, 1].")

    parquet_path = Path(args.parquet)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku Parquet: {parquet_path}")
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else parquet_path.with_name(DEFAULT_MANIFEST)
    )
    feature_names, feature_source = load_feature_names(
        manifest_path if not args.feature_columns else None,
        args.feature_columns,
        args.expected_feature_count,
    )

    LOG.info("Parquet: %s", parquet_path)
    LOG.info("Feature contract: %s (%d cech)", feature_source, len(feature_names))
    LOG.info("Wczytywanie Parquet V3...")
    df = pd.read_parquet(parquet_path)

    X, y, groups, dataset_summary, nan_report, sequence_column, sequence_kind = validate_and_prepare_data(
        df=df,
        feature_names=feature_names,
        target_column=args.target_column,
        match_column=args.match_column,
        max_nan_ratio=args.max_nan_ratio,
        frame_column=args.frame_column,
    )
    dataset_summary["parquet_path"] = str(parquet_path.resolve())
    dataset_summary["feature_manifest_path"] = str(manifest_path.resolve()) if manifest_path.exists() else None
    dataset_summary["feature_manifest_source"] = feature_source
    dataset_summary["feature_names"] = feature_names
    dataset_summary["fps_for_temporal_metrics"] = args.fps

    LOG.info(
        "Data Contract OK | rows=%d | matches=%d | features=%d | H=%d (%.1f%%) | A=%d (%.1f%%) | NaN=%.2f%%",
        dataset_summary["rows"],
        dataset_summary["matches"],
        dataset_summary["feature_count"],
        dataset_summary["class_counts"]["H_0"],
        100 * dataset_summary["class_shares"]["H_0"],
        dataset_summary["class_counts"]["A_1"],
        100 * dataset_summary["class_shares"]["A_1"],
        100 * dataset_summary["nan_ratio_overall"],
    )
    LOG.info("Temporal key: %s (%s)", sequence_column or "Parquet row order", sequence_kind or "row_order")

    cv_summary, oof_probs, oof_preds, fold_reports, model_reports = run_grouped_oof(
        X=X,
        y=y,
        groups=groups,
        n_splits=args.n_splits,
        random_state=args.random_state,
        df_for_sequence=df.reset_index(drop=True),
        sequence_column=sequence_column,
        sequence_kind=sequence_kind,
        fps=args.fps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nan_report.sort_values("nan_ratio", ascending=False).to_csv(
        output_dir / "role4_feature_nan_report.csv", index=False
    )

    # Persist OOF predictions so the team can inspect/calibrate only on the
    # external matches, without involving the target match.
    oof_table = pd.DataFrame(
        {
            args.match_column: groups,
            args.target_column: y,
        }
    )
    if sequence_column is not None:
        oof_table[sequence_column] = df[sequence_column].reset_index(drop=True)
    for model_name in oof_probs:
        oof_table[f"{model_name}_p_away"] = oof_probs[model_name]
        oof_table[f"{model_name}_p_home"] = 1.0 - oof_probs[model_name]
        oof_table[f"{model_name}_pred"] = oof_preds[model_name]
    oof_table.to_csv(output_dir / "role4_oof_predictions.csv", index=False)

    report = {
        "project": "FC Barcelona x Metrica Sports — More than a Hack 2027",
        "role": "Role 4 — H/A possession probabilities",
        "training_data_scope": "external SkillCorner Open Data only",
        "target_mapping": {"H": 0, "A": 1},
        "dataset": dataset_summary,
        "cross_validation": cv_summary,
        "models": model_reports,
        "important_notes": [
            "Metryki liczone na predykcjach out-of-fold; każdy fold walidacyjny zawiera całe mecze.",
            "Nie zastosowano losowego podziału klatek ani tuningu na docelowym meczu.",
            "HistGradientBoostingClassifier wykorzystuje natywną obsługę NaN; LR używa median imputation.",
            "Transition Count Ratio to proxy migotania/spell stability, a nie oficjalna metryka spellF1.",
            "Końcowe artefakty są fitowane wyłącznie na zewnętrznym zbiorze SkillCorner.",
        ],
    }
    report_path = output_dir / "role4_training_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, indent=2, ensure_ascii=False)

    LOG.info("\n=== OOF METRICS ===")
    for model_name, metrics in model_reports.items():
        frame = metrics["frame_level_oof"]
        temporal = metrics["temporal_stability_proxy"]
        ratio = temporal["transition_count_ratio_predicted_over_true"]
        ratio_display = "undefined (GT transitions=0)" if ratio is None else f"{ratio:.3f}"
        LOG.info(
            "%s | Precision_macro=%.4f Recall_macro=%.4f F1_macro=%.4f | "
            "F1_H=%.4f F1_A=%.4f | transitions pred/true=%d/%d | TCR=%s | mean_match_F1=%.4f",
            model_name,
            frame["precision_macro"],
            frame["recall_macro"],
            frame["f1_macro"],
            frame["H_0"]["f1"],
            frame["A_1"]["f1"],
            temporal["predicted_transition_count"],
            temporal["true_transition_count"],
            ratio_display,
            metrics["equal_weight_mean_match_f1_macro"],
        )

    LOG.info("Fit końcowych estimatorów na wszystkich zewnętrznych meczach SkillCorner...")
    final_estimators = train_final_estimators(X, y, args.random_state)
    for model_name, estimator in final_estimators.items():
        artifact = {
            "estimator": estimator,
            "feature_names": feature_names,
            "target_mapping": {"H": 0, "A": 1},
            "positive_class": 1,
            "training_source": "SkillCorner Open Data — external matches only",
            "fps_training_data": args.fps,
            "model_name": model_name,
        }
        artifact_path = output_dir / f"role4_{model_name}.joblib"
        joblib.dump(artifact, artifact_path)
        LOG.info("Zapisano: %s", artifact_path)

    LOG.info("Raport: %s", report_path)
    LOG.info("OOF predictions: %s", output_dir / "role4_oof_predictions.csv")
    LOG.info("Feature NaN report: %s", output_dir / "role4_feature_nan_report.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOG.exception("Role 4 training przerwany: %s", exc)
        raise
