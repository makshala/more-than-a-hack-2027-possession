"""Student Model: features + soft pseudo-labels -> P(H | Live), P(A | Live).

Stage 5. Consumes ``SoftLabels`` (from label_model.py) and returns
``LiveProbabilities`` (consumed by fusion_export.py) for every frame, including
frames where all LFs abstained: generalising beyond LF coverage is the purpose
of the student.

Two candidates are compared (CONFIG["student_model"]["models_to_compare"]):
* ``LogisticStudent`` - thin sklearn plumbing; every parameter from CONFIG.
* ``GAMStudent``      - interface skeleton only (library not chosen yet).

Soft targets are handled by duplicating each covered row as class H and class A
with weights p_H and p_A (``SoftLabels.expand_to_weighted_hard``), equivalent
to cross-entropy with soft targets.

Known risk (inform ablations): the student's features overlap with the inputs
of the LFs, so it may largely re-learn LF rules. Feature-group ablations
(``iter_group_ablations``) are the tool to measure that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from config import CONFIG
from contracts import (
    ContractError,
    LiveProbabilities,
    SoftLabels,
    same_frame_ids,
)
from feature_engineering import (
    CONTEXT_COLUMNS,
    FEATURE_GROUPS,
    IDENTIFIER_COLUMNS,
    NON_CAUSAL_FEATURES,
)


class StudentConfigError(ValueError):
    """Student configuration is missing or invalid."""


# ---------------------------------------------------------------------------
# Feature selection (config-driven, group-aware, causality-aware)
# ---------------------------------------------------------------------------
def select_feature_columns(
    features: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Resolve the predictor columns from CONFIG["student_model"]["features"].

    Order follows FEATURE_GROUPS. Fails (rather than guesses) when the group
    selection or causality policy is unset, when requested columns are
    missing, or when a selected column is not numeric (e.g. ``ball_status``
    is a string: exclude it or add an explicit encoding decision).
    """
    root = CONFIG if config is None else config
    fcfg = root["student_model"]["features"]

    groups = fcfg["groups"]
    if groups is None:
        raise StudentConfigError("student_model.features.groups is unset (use group names or 'all_predictors').")
    if groups == "all_predictors":
        group_names = tuple(FEATURE_GROUPS)
    else:
        unknown = [g for g in groups if g not in FEATURE_GROUPS]
        if unknown:
            raise StudentConfigError(f"Unknown feature groups: {unknown}")
        group_names = tuple(groups)

    policy = fcfg["causality_policy"]
    if policy not in ("causal_only", "allow_non_causal"):
        raise StudentConfigError("student_model.features.causality_policy must be 'causal_only' or 'allow_non_causal'.")

    cols: list[str] = [c for g in group_names for c in FEATURE_GROUPS[g]]
    cols += [c for c in fcfg["include_columns"] if c not in cols]
    if fcfg["exclude_identifier_and_context"]:
        blocked = set(IDENTIFIER_COLUMNS) | set(CONTEXT_COLUMNS)
        cols = [c for c in cols if c not in blocked]
    if policy == "causal_only":
        cols = [c for c in cols if c not in NON_CAUSAL_FEATURES]
    excluded = set(fcfg["exclude_columns"])
    cols = [c for c in cols if c not in excluded]

    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise KeyError(f"Selected predictors missing from features: {missing}")
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise StudentConfigError(
            f"Selected predictors are not numeric: {non_numeric}. Exclude them or add an encoding step."
        )
    if not cols:
        raise StudentConfigError("Feature selection produced no predictors.")
    return tuple(cols)


def iter_group_ablations(config: Mapping[str, Any] | None = None) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Yield ``(label, groups)``; leave-one-group-out over the configured groups."""
    root = CONFIG if config is None else config
    scheme = root["student_model"]["ablation"]["scheme"]
    if scheme is None:
        raise StudentConfigError("student_model.ablation.scheme is unset.")
    if scheme != "leave_one_group_out":
        raise StudentConfigError(f"Unsupported ablation scheme {scheme!r}")
    groups = root["student_model"]["features"]["groups"]
    if groups is None:
        raise StudentConfigError("student_model.features.groups is unset.")
    names = tuple(FEATURE_GROUPS) if groups == "all_predictors" else tuple(groups)
    for dropped in names:
        yield f"without:{dropped}", tuple(g for g in names if g != dropped)


def make_blocked_splits(frame_id: np.ndarray, config: Mapping[str, Any] | None = None):
    """Blocked temporal CV splits with purging. TODO (strategy from CONFIG)."""
    raise NotImplementedError("TODO: blocked temporal splits with purge_frames/n_blocks")


# ---------------------------------------------------------------------------
# Student interface
# ---------------------------------------------------------------------------
class StudentModel(ABC):
    """Common interface for all student candidates."""

    name: str = ""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._root = CONFIG if config is None else config
        self.config = self._root["student_model"]
        self._frame_id_column = self._root["fusion_export"]["frame_id_column"]
        self.feature_names_: tuple[str, ...] | None = None

    # -- shared input handling ------------------------------------------------
    def _predictors(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise RuntimeError("Model is not fitted.")
        X = features[list(self.feature_names_)]
        policy = self.config["features"]["missing_value_policy"]
        if policy is None:
            raise StudentConfigError("student_model.features.missing_value_policy is unset.")
        if policy != "error":
            raise NotImplementedError(f"TODO: missing_value_policy={policy!r}")
        if X.isna().any().any():
            bad = X.columns[X.isna().any()].tolist()
            raise ContractError(f"NaN in predictors under missing_value_policy='error': {bad}")
        return X

    def _check_alignment(self, features: pd.DataFrame, soft: SoftLabels) -> None:
        if self._frame_id_column not in features.columns:
            raise KeyError(f"features must contain {self._frame_id_column!r}")
        if not same_frame_ids(features[self._frame_id_column].to_numpy(), soft.frame_id):
            raise ContractError("features and soft labels must have identical frame_id order")

    # -- contract -------------------------------------------------------------
    @abstractmethod
    def fit(
        self,
        features: pd.DataFrame,
        soft_labels: SoftLabels,
        *,
        sample_weight: np.ndarray | None = None,
        train_mask: np.ndarray | None = None,
    ) -> "StudentModel":
        """Fit on covered rows of ``soft_labels``; ``features`` must be aligned."""

    @abstractmethod
    def predict_live(self, features: pd.DataFrame) -> LiveProbabilities:
        """P(H | Live), P(A | Live) for every row of ``features``."""

    def explain(self) -> pd.DataFrame:
        """Interpretability table (coefficients / smooth terms). TODO per model."""
        raise NotImplementedError


class LogisticStudent(StudentModel):
    """Regularised logistic regression on soft labels (sklearn)."""

    name = "logistic"
    _REQUIRED = ("C", "l1_ratio", "standardize")

    def _check_params(self) -> Mapping[str, Any]:
        params = self.config["logistic"]
        missing = [k for k in self._REQUIRED if params.get(k) is None]
        if missing:
            raise StudentConfigError(
                "student_model.logistic parameters still to be chosen: " + ", ".join(missing)
            )
        return params

    def fit(self, features, soft_labels, *, sample_weight=None, train_mask=None) -> "LogisticStudent":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        params = self._check_params()
        self._check_alignment(features, soft_labels)
        self.feature_names_ = select_feature_columns(features, self._root)
        X = self._predictors(features)

        row_index, y, weight = soft_labels.expand_to_weighted_hard(sample_weight, train_mask)
        if len(row_index) == 0:
            raise ContractError("No covered rows available for training.")

        kwargs: dict[str, Any] = {"C": float(params["C"]), "l1_ratio": float(params["l1_ratio"])}
        for key in ("solver", "max_iter", "class_weight"):
            if params.get(key) is not None:
                kwargs[key] = params[key]
        if self.config.get("random_state") is not None:
            kwargs["random_state"] = self.config["random_state"]

        steps = ([StandardScaler()] if params["standardize"] else []) + [LogisticRegression(**kwargs)]
        pipe = make_pipeline(*steps)
        X_train = X.to_numpy(dtype="float64")[row_index]
        pipe.fit(X_train, y, **{f"{pipe.steps[-1][0]}__sample_weight": weight})
        self._pipeline = pipe
        return self

    def predict_live(self, features: pd.DataFrame) -> LiveProbabilities:
        X = self._predictors(features)
        probs = self._pipeline.predict_proba(X.to_numpy(dtype="float64"))
        classes = list(self._pipeline.classes_)
        if classes != [0, 1]:  # H=0, A=1; both must have appeared in training
            raise ContractError(f"Student saw classes {classes}; both H and A are required in training.")
        return LiveProbabilities(
            frame_id=features[self._frame_id_column].to_numpy(),
            probs=probs,
            model_name=self.name,
        )

    def explain(self) -> pd.DataFrame:
        est = self._pipeline[-1]
        return pd.DataFrame({"coef": est.coef_[0]}, index=pd.Index(self.feature_names_, name="feature"))


class GAMStudent(StudentModel):
    """GAM student skeleton. Library, splines, smoothing and interactions TODO."""

    name = "gam"

    def _check_params(self) -> Mapping[str, Any]:
        params = self.config["gam"]
        missing = [k for k in ("library", "n_splines", "smoothing") if params.get(k) is None]
        if missing:
            raise StudentConfigError("student_model.gam parameters still to be chosen: " + ", ".join(missing))
        return params

    def fit(self, features, soft_labels, *, sample_weight=None, train_mask=None) -> "GAMStudent":
        self._check_params()
        self._check_alignment(features, soft_labels)
        raise NotImplementedError("TODO: GAM fit (library not chosen)")

    def predict_live(self, features: pd.DataFrame) -> LiveProbabilities:
        raise NotImplementedError("TODO: GAM predict")


STUDENT_REGISTRY: dict[str, type[StudentModel]] = {"logistic": LogisticStudent, "gam": GAMStudent}


def build_student(name: str, config: Mapping[str, Any] | None = None) -> StudentModel:
    if name not in STUDENT_REGISTRY:
        raise StudentConfigError(f"Unknown student model {name!r}; known: {sorted(STUDENT_REGISTRY)}")
    return STUDENT_REGISTRY[name](config)
