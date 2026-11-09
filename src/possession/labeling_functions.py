"""Labeling Functions (weak-supervision experts) for the possession pipeline.

Stage 3 of the pipeline. Each expert votes H, A or ABSTAIN per frame; it never
emits a probability. Votes are collected into a ``LabelMatrix`` (contracts.py).

Design notes
------------
* One class per expert, one shared base class with the generic plumbing:
  input checks, causality policy, quality gating (low quality -> ABSTAIN),
  output validation. Experts only implement ``_vote``.
* Three different notions must stay separate (see project brief):
  (1) observation quality  -> q_* columns from Quality Gating, used ONLY to
      abstain, never to pick H/A;
  (2) freshness of historical information -> part of an expert's own logic
      (e.g. ``frames_since_last_control`` in the history expert);
  (3) strength of evidence for possession -> not representable in a discrete
      vote. If needed, split an expert into several LFs (one per evidence
      level) rather than encoding strength in the vote.
* Experts are NOT assumed independent. ``correlation_group`` is metadata that
  the Label Model stage can use for the dependency hierarchy.
* Skeleton status: every concrete ``_vote`` raises NotImplementedError.
  No thresholds or rules are chosen here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config import CONFIG
from contracts import (
    ABSTAIN_CODE,
    VALID_VOTE_CODES,
    ContractError,
    LabelMatrix,
)
from feature_engineering import FEATURE_GROUPS, NON_CAUSAL_FEATURES


class LFConfigError(ValueError):
    """Required Labeling Function configuration is missing or invalid."""


@dataclass(frozen=True)
class LFResult:
    """Output of one LF: int8 votes (n,) and optional debug columns."""

    votes: np.ndarray
    debug: pd.DataFrame | None = None


class LabelingFunction(ABC):
    """Base class: one weak-supervision expert.

    Subclasses set the class attributes below and implement ``_vote``.

    Attributes
    ----------
    expert:
        Key in CONFIG["labeling_functions"]["experts"].
    source_groups:
        Names from ``FEATURE_GROUPS`` the expert reads. TODO: narrow the
        derived ``required_columns`` to the columns actually used.
    required_params:
        Names that must be set (not None) in
        CONFIG["labeling_functions"]["params"][expert] before the LF may run.
    correlation_group:
        Optional label for the dependency hierarchy in the Label Model.
    """

    expert: str = ""
    source_groups: tuple[str, ...] = ()
    required_params: tuple[str, ...] = ()
    correlation_group: str | None = None

    def __init__(self, config: Mapping[str, Any] | None = None, *, name: str | None = None) -> None:
        if not self.expert:
            raise TypeError(f"{type(self).__name__} must define the class attribute 'expert'")
        self._root = CONFIG if config is None else config
        self._lf_config: Mapping[str, Any] = self._root["labeling_functions"]
        if self.expert not in self._lf_config["experts"]:
            raise LFConfigError(f"expert {self.expert!r} is not listed in CONFIG experts")
        self.name = name or self.expert

    # ------------------------------------------------------------------ meta
    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(c for g in self.source_groups for c in FEATURE_GROUPS[g])

    @property
    def non_causal_columns(self) -> tuple[str, ...]:
        """Required columns that use future information (see feature_engineering)."""
        return tuple(c for c in self.required_columns if c in NON_CAUSAL_FEATURES)

    @property
    def quality_columns(self) -> tuple[str, ...]:
        return tuple(self._lf_config["quality_columns"].get(self.expert, ()))

    @property
    def params(self) -> Mapping[str, Any]:
        return self._lf_config["params"].get(self.expert, {})

    @property
    def quality_cutoff(self) -> float | None:
        return self._lf_config["quality_cutoffs"].get(self.expert)

    # ------------------------------------------------------------ validation
    def check_ready(self) -> None:
        """Fail early, with a readable list, if configuration is incomplete."""
        problems: list[str] = []
        cutoff = self.quality_cutoff
        if cutoff is None:
            problems.append(f"labeling_functions.quality_cutoffs[{self.expert!r}]")
        elif not (np.isfinite(float(cutoff)) and 0.0 <= float(cutoff) <= 1.0):
            raise LFConfigError(f"quality cutoff for {self.expert!r} must be in [0, 1]; got {cutoff!r}")
        problems += [
            f"labeling_functions.params[{self.expert!r}][{p!r}]"
            for p in self.required_params
            if self.params.get(p) is None
        ]
        policy = self._lf_config.get("causality_policy")
        if policy not in ("causal_only", "allow_non_causal"):
            problems.append("labeling_functions.causality_policy")
        if problems:
            raise LFConfigError(
                f"LF {self.name!r} cannot run; parameters still to be chosen: " + ", ".join(problems)
            )
        if policy == "causal_only" and self.non_causal_columns:
            raise LFConfigError(
                f"LF {self.name!r} reads non-causal columns {self.non_causal_columns} "
                "but causality_policy='causal_only'."
            )

    def check_inputs(self, features: pd.DataFrame, quality: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in features.columns]
        if missing:
            raise KeyError(f"LF {self.name!r} is missing feature columns: {missing}")
        missing_q = [c for c in self.quality_columns if c not in quality.columns]
        if missing_q:
            raise KeyError(f"LF {self.name!r} is missing quality columns: {missing_q}")
        if not features.index.equals(quality.index):
            raise ContractError("features and quality must share the same index and row order")

    # ------------------------------------------------------------------- API
    @abstractmethod
    def _vote(self, features: pd.DataFrame, quality: pd.DataFrame) -> LFResult:
        """Return raw votes (before quality gating). TODO in each expert."""

    def apply(self, features: pd.DataFrame, quality: pd.DataFrame) -> LFResult:
        """Run the expert, then abstain wherever observation quality is too low."""
        self.check_ready()
        self.check_inputs(features, quality)
        raw = self._vote(features, quality)
        votes = np.asarray(raw.votes)
        if votes.shape != (len(features),):
            raise ContractError(f"LF {self.name!r} returned votes of shape {votes.shape}")
        bad = set(np.unique(votes).tolist()) - VALID_VOTE_CODES
        if bad:
            raise ContractError(f"LF {self.name!r} returned invalid vote codes {sorted(bad)}")
        votes = votes.astype("int8", copy=True)

        if self.quality_columns:
            low = (quality[list(self.quality_columns)].to_numpy(dtype="float64")
                   < float(self.quality_cutoff)).any(axis=1)
            votes[low] = ABSTAIN_CODE

        debug = None
        if self._lf_config.get("emit_debug_columns", False) and raw.debug is not None:
            debug = raw.debug.add_prefix(f"{self.name}__")
        return LFResult(votes=votes, debug=debug)


# ---------------------------------------------------------------------------
# Experts (skeletons)
# ---------------------------------------------------------------------------
class PossessionHistoryLF(LabelingFunction):
    """Possession History Expert.

    Candidate evidence: last controlling team, its age, recent touch counts.
    TODO: rule, age/freshness handling (separate from q_history), parameters.
    """

    expert = "possession_history"
    source_groups = ("possession_history_and_control_windows",)
    correlation_group = "history"

    def _vote(self, features: pd.DataFrame, quality: pd.DataFrame) -> LFResult:
        raise NotImplementedError("TODO: Possession History Expert rule")


class GeometryLF(LabelingFunction):
    """Geometry Expert: ball-player proximity and pitch control.

    TODO: rule, parameters. Note the strong redundancy inside this group
    (see POTENTIAL_REDUNDANCIES); consider splitting into separate LFs only
    after correlation analysis.
    """

    expert = "geometry"
    source_groups = ("ball_player_proximity_and_pitch_control",)
    correlation_group = "geometry"

    def _vote(self, features: pd.DataFrame, quality: pd.DataFrame) -> LFResult:
        raise NotImplementedError("TODO: Geometry Expert rule")


class VideoTagLF(LabelingFunction):
    """Video Tag Expert: set pieces and restart team.

    TODO: rule, which tags imply a restart-team vote (use
    ``contracts.team_code_to_vote`` for team-coded columns).
    """

    expert = "video_tags"
    source_groups = ("video_tags_and_set_piece_context",)
    correlation_group = "tags"

    def _vote(self, features: pd.DataFrame, quality: pd.DataFrame) -> LFResult:
        raise NotImplementedError("TODO: Video Tag Expert rule")


class GoalkeeperLF(LabelingFunction):
    """Goalkeeper Expert: goalkeeper context and holding proxy. TODO: rule."""

    expert = "goalkeeper"
    source_groups = ("goalkeeper_context",)
    correlation_group = "goalkeeper"

    def _vote(self, features: pd.DataFrame, quality: pd.DataFrame) -> LFResult:
        raise NotImplementedError("TODO: Goalkeeper Expert rule")


class FormationLF(LabelingFunction):
    """Formation Expert: team shape and dynamics. TODO: rule."""

    expert = "formation"
    source_groups = ("formation_and_team_dynamics",)
    correlation_group = "formation"

    def _vote(self, features: pd.DataFrame, quality: pd.DataFrame) -> LFResult:
        raise NotImplementedError("TODO: Formation Expert rule")


EXPERT_REGISTRY: dict[str, type[LabelingFunction]] = {
    cls.expert: cls
    for cls in (PossessionHistoryLF, GeometryLF, VideoTagLF, GoalkeeperLF, FormationLF)
}


# ---------------------------------------------------------------------------
# Construction and label-matrix assembly
# ---------------------------------------------------------------------------
def build_labeling_functions(
    config: Mapping[str, Any] | None = None,
    experts: Sequence[str] | None = None,
) -> list[LabelingFunction]:
    """Instantiate the configured experts (in config order unless overridden)."""
    root = CONFIG if config is None else config
    names = tuple(root["labeling_functions"]["experts"]) if experts is None else tuple(experts)
    unknown = [n for n in names if n not in EXPERT_REGISTRY]
    if unknown:
        raise LFConfigError(f"No LF class registered for experts: {unknown}")
    return [EXPERT_REGISTRY[n](root) for n in names]


def apply_labeling_functions(
    features: pd.DataFrame,
    quality: pd.DataFrame,
    lfs: Sequence[LabelingFunction],
    *,
    frame_id_column: str | None = None,
) -> tuple[LabelMatrix, pd.DataFrame | None]:
    """Run all LFs and assemble the Label Matrix.

    Returns ``(label_matrix, debug_frame_or_None)``. Row order is preserved.
    ``frame_id`` is read from ``frame_id_column`` (default: the config's
    fusion_export frame_id column).
    """
    if not lfs:
        raise ValueError("At least one labeling function is required.")
    column = frame_id_column or CONFIG["fusion_export"]["frame_id_column"]
    if column not in features.columns:
        raise KeyError(f"features must contain {column!r}")

    results = [lf.apply(features, quality) for lf in lfs]
    values = np.column_stack([r.votes for r in results]).astype("int8")
    matrix = LabelMatrix(values, tuple(lf.name for lf in lfs), features[column].to_numpy())

    debug_parts = [r.debug for r in results if r.debug is not None]
    debug = pd.concat(debug_parts, axis=1) if debug_parts else None
    return matrix, debug
