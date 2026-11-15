"""Generative Label Model: Label Matrix -> soft pseudo-labels.

Stage 4. Input is a ``LabelMatrix``; output is ``SoftLabels`` (contracts.py),
which is exactly what the Student Model consumes.

Backends
--------
* ``snorkel``       adapter skeleton around Snorkel's LabelModel (reference
                    method). Optional dependency, imported lazily. NOT
                    implemented yet and NOT tested in this skeleton.
* ``majority_vote`` plain vote share. A reference baseline that makes the
                    contract executable without extra dependencies. It treats
                    all LFs as equally reliable and independent, which is
                    exactly the assumption we do not want to rely on.

Cautions carried over from the project brief
--------------------------------------------
* Posteriors are NOT calibrated probabilities (no independent Ground Truth).
* LFs are not independent. ``dependency_groups`` in CONFIG will define the
  hierarchy; how a backend uses it is a TODO decided after LF analysis.
  Note (verify against the installed version): Snorkel 0.9.x LabelModel
  assumes conditionally independent LFs and, to my knowledge, has no argument
  for an explicit dependency structure. Hierarchical merging of correlated
  LFs before fitting may therefore be the practical route.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config import CONFIG
from contracts import (
    ABSTAIN_CODE,
    AWAY_CODE,
    HOME_CODE,
    ContractError,
    LabelMatrix,
    SoftLabels,
    validate_class_order,
    HA_ORDER,
)


class LabelModelConfigError(ValueError):
    """Label Model configuration is missing or invalid."""


# ---------------------------------------------------------------------------
# Descriptive diagnostics (counting only; no modelling)
# ---------------------------------------------------------------------------
def lf_summary(matrix: LabelMatrix) -> pd.DataFrame:
    """Per-LF coverage, overlap, conflict and vote balance.

    coverage: share of frames where the LF votes.
    overlap:  share where the LF and at least one other LF vote.
    conflict: share where the LF votes and another LF votes differently.
    """
    v = matrix.values
    votes = v != ABSTAIN_CODE
    n_votes = votes.sum(axis=1, keepdims=True)
    h = (v == HOME_CODE).astype(int)
    a = (v == AWAY_CODE).astype(int)
    h_others = h.sum(axis=1, keepdims=True) - h
    a_others = a.sum(axis=1, keepdims=True) - a
    overlap = votes & ((n_votes - votes.astype(int)) > 0)
    conflict = ((h == 1) & (a_others > 0)) | ((a == 1) & (h_others > 0))
    n = max(matrix.n_frames, 1)
    return pd.DataFrame(
        {
            "coverage": votes.sum(axis=0) / n,
            "overlap": overlap.sum(axis=0) / n,
            "conflict": conflict.sum(axis=0) / n,
            "share_home_votes": np.where(votes.sum(axis=0) > 0, h.sum(axis=0) / np.maximum(votes.sum(axis=0), 1), np.nan),
        },
        index=pd.Index(matrix.lf_names, name="lf"),
    )


def pairwise_agreement(matrix: LabelMatrix) -> pd.DataFrame:
    """Agreement rate on frames where both LFs vote (NaN if never both vote).

    High pairwise agreement/overlap is a prompt to inspect LF dependence; it
    is not by itself a proof of redundancy or of correctness.
    """
    v = matrix.values
    m = matrix.n_lfs
    out = np.full((m, m), np.nan)
    for i in range(m):
        for j in range(m):
            both = (v[:, i] != ABSTAIN_CODE) & (v[:, j] != ABSTAIN_CODE)
            if both.any():
                out[i, j] = float((v[both, i] == v[both, j]).mean())
    idx = pd.Index(matrix.lf_names, name="lf")
    return pd.DataFrame(out, index=idx, columns=idx)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class LabelModelBackend(ABC):
    """Backend contract. ``predict_proba`` returns (n, 2) in (H, A) order.

    Rows where every LF abstains may hold any finite placeholder; the wrapper
    marks them ``covered=False``.
    """

    name: str = ""

    @abstractmethod
    def fit(self, matrix: LabelMatrix, *, train_mask: np.ndarray | None = None) -> "LabelModelBackend":
        ...

    @abstractmethod
    def predict_proba(self, matrix: LabelMatrix) -> np.ndarray:
        ...


class MajorityVoteBackend(LabelModelBackend):
    """Reference baseline: share of H among non-abstaining votes."""

    name = "majority_vote"

    def fit(self, matrix: LabelMatrix, *, train_mask: np.ndarray | None = None) -> "MajorityVoteBackend":
        self._lf_names = matrix.lf_names  # nothing to learn
        return self

    def predict_proba(self, matrix: LabelMatrix) -> np.ndarray:
        if matrix.lf_names != self._lf_names:
            raise ContractError("LF set differs between fit and predict")
        v = matrix.values
        n_h = (v == HOME_CODE).sum(axis=1).astype("float64")
        n_a = (v == AWAY_CODE).sum(axis=1).astype("float64")
        total = n_h + n_a
        p_h = np.where(total > 0, n_h / np.where(total > 0, total, 1.0), 0.5)  # 0.5 = placeholder
        return np.column_stack([p_h, 1.0 - p_h])


class SnorkelBackend(LabelModelBackend):
    """Adapter skeleton for ``snorkel.labeling.model.LabelModel``.

    TODO: fit/predict. Hyperparameters come from CONFIG["label_model"]["snorkel"];
    none are defaulted. Snorkel's label convention (-1 abstain, 0..k-1 classes)
    already matches ``LabelMatrix``. Check the installed Snorkel version's
    compatibility with the project's pandas/numpy before adopting it.
    """

    name = "snorkel"

    def __init__(self, params: Mapping[str, Any]) -> None:
        missing = [k for k in ("n_epochs", "lr", "l2", "seed") if params.get(k) is None]
        if missing:
            raise LabelModelConfigError(
                "label_model.snorkel parameters still to be chosen: " + ", ".join(missing)
            )
        try:
            from snorkel.labeling.model import LabelModel  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'snorkel' backend needs the optional dependency 'snorkel' "
                "(e.g. `uv add snorkel`); check compatibility with the pinned pandas/numpy."
            ) from exc
        self._params = dict(params)

    def fit(self, matrix: LabelMatrix, *, train_mask: np.ndarray | None = None) -> "SnorkelBackend":
        raise NotImplementedError("TODO: Snorkel LabelModel fit")

    def predict_proba(self, matrix: LabelMatrix) -> np.ndarray:
        raise NotImplementedError("TODO: Snorkel LabelModel predict_proba")


def build_backend(config: Mapping[str, Any] | None = None) -> LabelModelBackend:
    cfg = (CONFIG if config is None else config)["label_model"]
    name = cfg["backend"]
    if name == "majority_vote":
        return MajorityVoteBackend()
    if name == "snorkel":
        return SnorkelBackend(cfg["snorkel"])
    raise LabelModelConfigError(f"Unknown label_model.backend {name!r}")


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------
@dataclass
class GenerativeLabelModel:
    """Facade: validates inputs, delegates to a backend, returns ``SoftLabels``."""

    config: Mapping[str, Any]
    backend: LabelModelBackend

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "GenerativeLabelModel":
        root = CONFIG if config is None else config
        validate_class_order(root["label_model"]["class_order"], HA_ORDER)
        return cls(config=root, backend=build_backend(root))

    def _check_dependency_groups(self, matrix: LabelMatrix) -> None:
        groups = self.config["label_model"]["dependency_groups"]
        unknown = sorted({lf for members in groups.values() for lf in members} - set(matrix.lf_names))
        if unknown:
            raise LabelModelConfigError(f"dependency_groups reference unknown LFs: {unknown}")
        # TODO: define how groups enter the model (merge, hierarchy, explicit deps).

    def fit(self, matrix: LabelMatrix, *, train_mask: np.ndarray | None = None) -> "GenerativeLabelModel":
        self._check_dependency_groups(matrix)
        if train_mask is not None and np.asarray(train_mask).shape != (matrix.n_frames,):
            raise ContractError("train_mask must have shape (n,)")
        # TODO: honour label_model.fit_on_time_blocks and training_row_filter.
        self.backend.fit(matrix, train_mask=train_mask)
        return self

    def predict_soft_labels(self, matrix: LabelMatrix) -> SoftLabels:
        probs = np.asarray(self.backend.predict_proba(matrix), dtype="float64")
        covered = (matrix.values != ABSTAIN_CODE).any(axis=1)
        return SoftLabels(
            frame_id=matrix.frame_id,
            probs=probs,
            covered=covered,
            backend=self.backend.name,
            metadata={"calibrated": False, "lf_names": matrix.lf_names},
        )

    def fit_predict(self, matrix: LabelMatrix, *, train_mask: np.ndarray | None = None) -> SoftLabels:
        return self.fit(matrix, train_mask=train_mask).predict_soft_labels(matrix)
