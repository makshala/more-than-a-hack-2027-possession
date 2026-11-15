"""Quality scores for the possession pipeline.

This module estimates *observation/source quality*, not ball possession.
It does not emit H/A/D labels and does not decide which team controls the ball.

Expected project usage:
    scores = compute_quality_scores(
        features,
        CONFIG["quality_gating"],
        include_components=True,
    )
    features_with_quality = append_quality_scores(features, scores)

All scale parameters and the interpretation of ball-tracking statuses are
explicit configuration choices. There are deliberately no guessed decay
thresholds in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


class QualityConfigError(ValueError):
    """Raised when required quality-gating configuration is missing/invalid."""


# ---------------------------------------------------------------------------
# Source columns used by each quality score
# ---------------------------------------------------------------------------
BALL_COLUMNS: tuple[str, ...] = (
    "ball_status",
    "ball_sigma_m",
    "ball_gap_frames",
    "ball_in_camera_fov",
)

PLAYER_COLUMNS: tuple[str, ...] = ("home_n_visible", "away_n_visible")
HOMOGRAPHY_COLUMNS: tuple[str, ...] = ("homography_reproj_err",)
HISTORY_COLUMNS: tuple[str, ...] = ("last_controlled_team", "frames_since_last_control")
PITCH_COLUMNS: tuple[str, ...] = ("home_pitch_control",)

TAG_BINARY_COLUMNS: tuple[str, ...] = (
    "st_set_piece",
    "st_throw_in",
    "st_corner_kick",
    "st_goal_kick",
    "st_free_kick",
    "st_kick_off",
    "st_goal",
    "st_post_goal_dead",
    "st_defending_tag",
    "st_in_box_tag",
)
TAG_TEAM_COLUMNS: tuple[str, ...] = ("st_restart_team", "st_team_prior")
TAG_NUMERIC_COLUMNS: tuple[str, ...] = ("st_team_score", "st_n_active_tags")
TAG_COLUMNS: tuple[str, ...] = TAG_BINARY_COLUMNS + TAG_TEAM_COLUMNS + TAG_NUMERIC_COLUMNS

FORMATION_NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "home_mean_speed",
    "away_mean_speed",
    "home_length_x",
    "away_length_x",
    "home_width_y",
    "away_width_y",
    "all_players_mean_speed",
    "players_in_penalty_box",
)

FORMATION_COLUMNS: tuple[str, ...] = (
    "home_centroid_x",
    "away_centroid_x",
    "home_centroid_y",
    "away_centroid_y",
    "home_centroid_vx",
    "away_centroid_vx",
    "home_centroid_vy",
    "away_centroid_vy",
    "home_mean_speed",
    "away_mean_speed",
    "home_length_x",
    "away_length_x",
    "home_width_y",
    "away_width_y",
    "all_players_mean_speed",
    "players_walking_ratio",
    "players_in_penalty_box",
    "home_centroid_x_norm",
    "away_centroid_x_norm",
    "collective_vx_norm",
)

BROADCAST_COLUMNS: tuple[str, ...] = (
    "brd_valid",
    "brd_is_replay",
    "frames_since_brd_replay",
    "brd_lookat_x",
    "brd_lookat_y",
    "brd_ball_x",
    "brd_ball_y",
)


# ---------------------------------------------------------------------------
# Small, reusable validation and scoring helpers
# ---------------------------------------------------------------------------
def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            "Quality Gating is missing required columns: "
            f"{missing}. Check the Feature Engineering output and schema version."
        )


def _required_number(config: Mapping[str, Any], key: str, section: str) -> float:
    value = config.get(key)
    if value is None:
        raise QualityConfigError(
            f"Missing CONFIG['quality_gating']['{section}']['{key}']. "
            "Choose this parameter explicitly; no threshold is inferred here."
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise QualityConfigError(
            f"{section}.{key} must be a finite positive number; got {value!r}."
        ) from exc
    if not np.isfinite(number) or number <= 0:
        raise QualityConfigError(
            f"{section}.{key} must be a finite positive number; got {value!r}."
        )
    return number


def _as_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _exp_decay(values: pd.Series, scale: float) -> pd.Series:
    """Map non-negative magnitudes to [0, 1] using exp(-x / scale)."""
    values = pd.to_numeric(values, errors="coerce").astype("float64")
    valid = values.notna() & np.isfinite(values) & values.ge(0)
    result = pd.Series(0.0, index=values.index, dtype="float64")
    result.loc[valid] = np.exp(-values.loc[valid] / scale)
    return result.clip(0.0, 1.0)


def _combine_components(
    components: Mapping[str, pd.Series],
    *,
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Combine [0, 1] quality components with a weighted geometric mean.

    If no weights are supplied, components receive equal weight. A zero
    component makes the combined score zero, which is a conservative gate.
    This is a heuristic quality index, not a calibrated probability.
    """
    if not components:
        raise ValueError("At least one component is required to combine quality scores.")

    names = tuple(components)
    if weights is None:
        weight_values = np.ones(len(names), dtype="float64")
    else:
        unknown = set(weights) - set(names)
        if unknown:
            raise QualityConfigError(f"Unknown component weight names: {sorted(unknown)}")
        weight_values = np.asarray([float(weights.get(name, 1.0)) for name in names])
        if not np.isfinite(weight_values).all() or (weight_values <= 0).any():
            raise QualityConfigError("Component weights must be finite and greater than zero.")

    weight_values = weight_values / weight_values.sum()
    matrix = pd.concat(
        [components[name].astype("float64").clip(0.0, 1.0) for name in names],
        axis=1,
    ).to_numpy(dtype="float64")
    # Avoid log(0); restore exact zero where any component is zero.
    has_zero = np.any(matrix <= 0.0, axis=1)
    log_values = np.log(np.clip(matrix, 1e-12, 1.0))
    combined = np.exp(log_values @ weight_values)
    combined[has_zero] = 0.0
    return pd.Series(combined, index=components[names[0]].index, dtype="float64")


def _binary_validity(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.isin((0, 1)).astype("float64")


def _team_code_validity(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.isin((-1, 0, 1)).astype("float64")


def _finite_validity(series: pd.Series, *, non_negative: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    valid = numeric.notna() & np.isfinite(numeric)
    if non_negative:
        valid &= numeric.ge(0)
    return valid.astype("float64")


def _component_weights(
    config: Mapping[str, Any],
    *,
    section: str,
) -> Mapping[str, float] | None:
    weights = config.get("component_weights")
    if weights is None:
        return None
    if not isinstance(weights, Mapping):
        raise QualityConfigError(f"{section}.component_weights must be a mapping.")
    return weights


# ---------------------------------------------------------------------------
# Per-source quality scores
# ---------------------------------------------------------------------------
def _ball_quality(frame: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, pd.Series]:
    section = "ball"
    status_quality = config.get("status_quality")
    fov_quality = config.get("fov_quality")
    if not isinstance(status_quality, Mapping) or not status_quality:
        raise QualityConfigError(
            "ball.status_quality must map OBSERVED, INTERPOLATED_SHORT and "
            "INFERRED_LATENT to explicit quality values in [0, 1]."
        )
    if not isinstance(fov_quality, Mapping) or not fov_quality:
        raise QualityConfigError("ball.fov_quality must map 0 and 1 to values in [0, 1].")

    try:
        status_scores = {
            str(key).strip().upper(): float(value)
            for key, value in status_quality.items()
        }
        fov_scores = {int(key): float(value) for key, value in fov_quality.items()}
    except (TypeError, ValueError) as exc:
        raise QualityConfigError(
            "ball.status_quality and ball.fov_quality values must be numeric in [0, 1]."
        ) from exc
    required_statuses = {"OBSERVED", "INTERPOLATED_SHORT", "INFERRED_LATENT"}
    missing_statuses = required_statuses - set(status_scores)
    if missing_statuses:
        raise QualityConfigError(
            "ball.status_quality must explicitly configure all tracking states; "
            f"missing {sorted(missing_statuses)}."
        )
    if not {0, 1}.issubset(fov_scores):
        raise QualityConfigError("ball.fov_quality must explicitly configure both 0 and 1.")
    for label, value in status_scores.items():
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise QualityConfigError(f"ball.status_quality[{label!r}] must be in [0, 1].")
    for label, value in fov_scores.items():
        if label not in (0, 1) or not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise QualityConfigError("ball.fov_quality must contain only 0/1 keys and [0, 1] values.")

    sigma_scale = _required_number(config, "sigma_scale_m", section)
    gap_scale = _required_number(config, "gap_scale_frames", section)

    status = frame["ball_status"].astype("string").str.strip().str.upper()
    q_status = status.map(status_scores).fillna(0.0).astype("float64")

    sigma = _as_numeric(frame, "ball_sigma_m")
    q_sigma = _exp_decay(sigma, sigma_scale)

    gap = _as_numeric(frame, "ball_gap_frames")
    # A null gap length for an explicitly OBSERVED ball means "no active gap",
    # rather than an unknown gap. For inferred/interpolated rows, null stays
    # unknown and therefore receives zero quality conservatively.
    gap = gap.mask(gap.isna() & status.eq("OBSERVED"), 0.0)
    q_gap = _exp_decay(gap, gap_scale)

    fov = pd.to_numeric(frame["ball_in_camera_fov"], errors="coerce")
    q_fov = fov.map(fov_scores).fillna(0.0).astype("float64")

    components = {
        "status": q_status,
        "sigma": q_sigma,
        "gap": q_gap,
        "field_of_view": q_fov,
    }
    return {
        "q_ball": _combine_components(
            components,
            weights=_component_weights(config, section=section),
        ),
        **{f"q_ball_{name}": value for name, value in components.items()},
    }


def _players_quality(frame: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, pd.Series]:
    expected = _required_number(config, "expected_players_per_team", "players")
    home = _as_numeric(frame, "home_n_visible")
    away = _as_numeric(frame, "away_n_visible")

    q_home = (home.clip(lower=0.0, upper=expected) / expected).where(
        home.notna() & np.isfinite(home) & home.ge(0), 0.0
    )
    q_away = (away.clip(lower=0.0, upper=expected) / expected).where(
        away.notna() & np.isfinite(away) & away.ge(0), 0.0
    )
    q_players = _combine_components(
        {"home": q_home, "away": q_away},
        weights=_component_weights(config, section="players"),
    )
    return {
        "q_players": q_players,
        "q_home_players": q_home.astype("float64"),
        "q_away_players": q_away.astype("float64"),
    }


def _homography_quality(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, pd.Series]:
    scale = _required_number(config, "reprojection_error_scale", "homography")
    error = _as_numeric(frame, "homography_reproj_err")
    q = _exp_decay(error, scale)
    return {"q_homography": q, "q_homography_error": q.copy()}


def _history_quality(frame: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, pd.Series]:
    decay = _required_number(config, "age_decay_frames", "history")
    last_team = _as_numeric(frame, "last_controlled_team")
    age = _as_numeric(frame, "frames_since_last_control")

    # A known last controlling team is required. Team code 0 means unknown.
    q_known = last_team.isin((-1, 1)).astype("float64")
    q_freshness = _exp_decay(age, decay)
    q = _combine_components(
        {"known_team": q_known, "freshness": q_freshness},
        weights=_component_weights(config, section="history"),
    )
    return {
        "q_history": q,
        "q_history_known_team": q_known,
        "q_history_freshness": q_freshness,
    }


def _tags_quality(frame: pd.DataFrame) -> dict[str, pd.Series]:
    validity: dict[str, pd.Series] = {}
    for column in TAG_BINARY_COLUMNS:
        validity[column] = _binary_validity(frame[column])
    for column in TAG_TEAM_COLUMNS:
        validity[column] = _team_code_validity(frame[column])
    validity["st_team_score"] = _finite_validity(frame["st_team_score"])
    validity["st_n_active_tags"] = _finite_validity(
        frame["st_n_active_tags"], non_negative=True
    )
    q = pd.concat(validity, axis=1).mean(axis=1).fillna(0.0).astype("float64")
    return {"q_tags": q, "q_tags_valid_fields": q.copy()}


def _formation_quality(
    frame: pd.DataFrame,
    q_players: pd.Series,
    config: Mapping[str, Any],
) -> dict[str, pd.Series]:
    validity_parts: dict[str, pd.Series] = {}
    for column in FORMATION_COLUMNS:
        validity_parts[column] = _finite_validity(
            frame[column],
            non_negative=column in FORMATION_NON_NEGATIVE_COLUMNS,
        )
    walking_ratio = _as_numeric(frame, "players_walking_ratio")
    validity_parts["players_walking_ratio_range"] = (
        walking_ratio.notna() & np.isfinite(walking_ratio) & walking_ratio.between(0.0, 1.0)
    ).astype("float64")
    validity = pd.concat(validity_parts, axis=1)
    q_fields = validity.mean(axis=1).fillna(0.0).astype("float64")
    # Formation summaries are less representative when too few players are
    # visible, even when all summary columns happen to contain numeric values.
    q = _combine_components(
        {"field_coverage": q_fields, "player_visibility": q_players},
        weights=_component_weights(config, section="formation"),
    )
    return {
        "q_formation": q,
        "q_formation_valid_fields": q_fields,
        "q_formation_player_visibility": q_players,
    }


def _pitch_quality(
    frame: pd.DataFrame,
    q_players: pd.Series,
    q_homography: pd.Series,
    config: Mapping[str, Any],
) -> dict[str, pd.Series]:
    pitch = _as_numeric(frame, "home_pitch_control")
    valid = pitch.notna() & np.isfinite(pitch) & pitch.between(0.0, 1.0)
    q_valid = valid.astype("float64")
    # Pitch-control estimates depend on player locations and pitch geometry.
    q = _combine_components(
        {
            "valid_pitch_control": q_valid,
            "player_visibility": q_players,
            "homography": q_homography,
        },
        weights=_component_weights(config, section="pitch"),
    )
    return {
        "q_pitch": q,
        "q_pitch_valid_value": q_valid,
        "q_pitch_players": q_players,
        "q_pitch_homography": q_homography,
    }


def _broadcast_quality(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, pd.Series]:
    recovery_scale = _required_number(config, "replay_recovery_frames", "broadcast")

    brd_valid = pd.to_numeric(frame["brd_valid"], errors="coerce")
    replay = pd.to_numeric(frame["brd_is_replay"], errors="coerce")
    q_valid = brd_valid.eq(1).astype("float64")
    q_not_replay = replay.eq(0).astype("float64")

    replay_age = _as_numeric(frame, "frames_since_brd_replay")
    valid_age = replay_age.notna() & np.isfinite(replay_age) & replay_age.ge(0)
    q_recovery = pd.Series(0.0, index=frame.index, dtype="float64")
    q_recovery.loc[valid_age] = 1.0 - np.exp(
        -replay_age.loc[valid_age] / recovery_scale
    )

    coordinate_columns = ("brd_lookat_x", "brd_lookat_y", "brd_ball_x", "brd_ball_y")
    q_coordinates = pd.concat(
        {column: _finite_validity(frame[column]) for column in coordinate_columns},
        axis=1,
    ).mean(axis=1).fillna(0.0)

    components = {
        "valid_pitch_view": q_valid,
        "not_replay": q_not_replay,
        "replay_recovery": q_recovery,
        "coordinates_available": q_coordinates,
    }
    return {
        "q_broadcast": _combine_components(
            components,
            weights=_component_weights(config, section="broadcast"),
        ),
        **{f"q_broadcast_{name}": value for name, value in components.items()},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def required_quality_columns(*, include_broadcast: bool = True) -> tuple[str, ...]:
    """Return all raw columns required by the configured quality groups."""
    groups: tuple[tuple[str, ...], ...] = (
        BALL_COLUMNS,
        PLAYER_COLUMNS,
        HOMOGRAPHY_COLUMNS,
        HISTORY_COLUMNS,
        TAG_COLUMNS,
        FORMATION_COLUMNS,
        PITCH_COLUMNS,
    )
    if include_broadcast:
        groups += (BROADCAST_COLUMNS,)
    return tuple(dict.fromkeys(column for group in groups for column in group))


def compute_quality_scores(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    include_components: bool = False,
    include_broadcast: bool = True,
) -> pd.DataFrame:
    """Compute per-frame source quality scores in [0, 1].

    Parameters
    ----------
    frame:
        Feature-engineering output. This function does not modify it.
    config:
        The ``CONFIG["quality_gating"]`` mapping from the project's global
        configuration. See the example block in the module docstring/docs.
    include_components:
        Include component scores (e.g. q_ball_sigma) for diagnostics.
    include_broadcast:
        Compute q_broadcast. Keep True when using the first Feature Engineering
        version that includes the TV/Broadcast columns.

    Returns
    -------
    pd.DataFrame
        A frame-aligned table with q_* columns. Scores are heuristic quality
        indices, not P(correct), class probabilities, or calibrated confidence.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping, normally CONFIG['quality_gating'].")

    required = required_quality_columns(include_broadcast=include_broadcast)
    _require_columns(frame, required)

    required_sections = {
        "ball": ("status_quality", "fov_quality", "sigma_scale_m", "gap_scale_frames"),
        "players": ("expected_players_per_team",),
        "homography": ("reprojection_error_scale",),
        "history": ("age_decay_frames",),
    }
    if include_broadcast:
        required_sections["broadcast"] = ("replay_recovery_frames",)
    for section, keys in required_sections.items():
        section_config = config.get(section)
        if not isinstance(section_config, Mapping):
            raise QualityConfigError(f"Missing CONFIG['quality_gating']['{section}'] mapping.")
        missing = [key for key in keys if key not in section_config or section_config[key] is None]
        if missing:
            raise QualityConfigError(
                f"Missing quality-gating parameters in section {section!r}: {missing}."
            )

    outputs: dict[str, pd.Series] = {}
    outputs.update(_ball_quality(frame, config["ball"]))
    outputs.update(_players_quality(frame, config["players"]))
    homography_outputs = _homography_quality(frame, config["homography"])
    outputs.update(homography_outputs)
    outputs.update(_history_quality(frame, config["history"]))
    outputs.update(_tags_quality(frame))
    if "formation" in config and not isinstance(config["formation"], Mapping):
        raise QualityConfigError("quality_gating.formation must be a mapping when provided.")
    if "pitch" in config and not isinstance(config["pitch"], Mapping):
        raise QualityConfigError("quality_gating.pitch must be a mapping when provided.")
    outputs.update(
        _formation_quality(frame, outputs["q_players"], config.get("formation", {}))
    )
    outputs.update(
        _pitch_quality(
            frame,
            outputs["q_players"],
            homography_outputs["q_homography"],
            config.get("pitch", {}),
        )
    )
    if include_broadcast:
        outputs.update(_broadcast_quality(frame, config["broadcast"]))

    scores = pd.DataFrame(outputs, index=frame.index)
    quality_columns = [column for column in scores.columns if column.startswith("q_")]
    values = scores[quality_columns].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise RuntimeError("Quality Gating produced NaN or infinite scores.")
    if (values < 0.0).any() or (values > 1.0).any():
        raise RuntimeError("Quality Gating produced scores outside [0, 1].")

    if not include_components:
        scores = scores[[
            "q_ball",
            "q_players",
            "q_homography",
            "q_history",
            "q_tags",
            "q_formation",
            "q_pitch",
            *(["q_broadcast"] if include_broadcast else []),
        ]]
    return scores


def append_quality_scores(
    frame: pd.DataFrame,
    quality_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Append aligned quality columns without modifying the source frame."""
    if not frame.index.equals(quality_scores.index):
        raise ValueError("frame and quality_scores must have identical indexes and row order.")
    collisions = sorted(set(frame.columns).intersection(quality_scores.columns))
    if collisions:
        raise ValueError(f"Refusing to overwrite existing columns: {collisions}")
    return pd.concat([frame.copy(), quality_scores.copy()], axis=1)
