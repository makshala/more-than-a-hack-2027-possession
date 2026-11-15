"""Feature registry and first-stage utilities for the possession pipeline.

This module intentionally does not fit a model or decide which input columns
should be discarded. It records the agreed 111-column schema, validates it,
and provides a small set of transparent candidate feature transformations.

The global project configuration remains the single source of runtime config;
no separate config file is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Agreed semantic groups
# ---------------------------------------------------------------------------
# Every input column is listed exactly once. `frame_id` is retained as the
# observation key, but must not be passed to the possession classifier.

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "frame_and_match_context": (
        "frame_id",
        "period",
        "is_graded",
        "is_pre_kickoff_d",
        "timestamp_s",
    ),
    "ball_tracking_motion_and_observability": (
        "ball_x",
        "ball_y",
        "ball_vx",
        "ball_vy",
        "ball_speed",
        "ball_acc",
        "ball_status",
        "ball_status_code",
        "ball_sigma_m",
        "ball_gap_frames",
        "frames_since_ball_gap",
        "frames_to_ball_gap",
        "ball_near_boundary",
        "ball_in_camera_fov",
        "ball_stationary_frames",
        "ball_deflection_angle",
        "ball_x_norm",
        "ball_vx_norm",
    ),
    "tactical_camera_and_homography": (
        "homography_reproj_err",
        "camera_pan_speed",
    ),
    "video_tags_and_set_piece_context": (
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
        "st_restart_team",
        "st_team_prior",
        "st_team_score",
        "st_n_active_tags",
    ),
    "player_visibility": (
        "home_n_visible",
        "away_n_visible",
    ),
    "formation_and_team_dynamics": (
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
    ),
    "ball_player_proximity_and_pitch_control": (
        "home_in_3m",
        "away_in_3m",
        "home_in_5m",
        "away_in_5m",
        "home_min_tti_s",
        "away_min_tti_s",
        "home_near1_dist_m",
        "away_near1_dist_m",
        "home_near1_speed_ms",
        "away_near1_speed_ms",
        "home_near2_dist_m",
        "away_near2_dist_m",
        "home_near2_speed_ms",
        "away_near2_speed_ms",
        "home_near3_dist_m",
        "away_near3_dist_m",
        "home_near3_speed_ms",
        "away_near3_speed_ms",
        "home_near1_closing_speed_ms",
        "away_near1_closing_speed_ms",
        "home_near1_vel_cos",
        "away_near1_vel_cos",
        "near1_dist_diff_m",
        "tti_diff_s",
        "home_pitch_control",
        "is_contested",
    ),
    "possession_history_and_control_windows": (
        "home_consecutive_touches",
        "away_consecutive_touches",
        "last_controlled_team",
        "frames_since_last_control",
        "control_diff_roll50",
        "control_diff_roll100",
        "control_diff_causal50",
        "control_diff_causal100",
    ),
    "goalkeeper_context": (
        "home_gk_dist_m",
        "away_gk_dist_m",
        "gk_holding_ball_proxy",
    ),
    "broadcast_camera_context": (
        "brd_valid",
        "brd_is_replay",
        "frames_since_brd_replay",
        "brd_lookat_x",
        "brd_lookat_y",
        "brd_lookat_speed_ms",
        "brd_ball_x",
        "brd_ball_y",
        "ball_to_brd_lookat_dist_m",
    ),
    "leadlag_transition_features": (
        "near1_diff_leadlag_10f",
        "near1_diff_leadlag_25f",
        "pitch_ctrl_leadlag_10f",
        "pitch_ctrl_leadlag_25f",
    ),
}


# Key/context columns are retained in the processed frame but excluded from
# the default predictor matrix. `period` and the flags may still be used for
# filtering, segmentation, or diagnostics.
IDENTIFIER_COLUMNS: tuple[str, ...] = ("frame_id",)
CONTEXT_COLUMNS: tuple[str, ...] = (
    "period",
    "is_graded",
    "is_pre_kickoff_d",
    "timestamp_s",
)

# These columns use future observations by definition or by their documented
# central/lead-lag windows. They are not removed; mark them explicitly so the
# team can decide whether offline look-ahead is permitted at inference time.
NON_CAUSAL_FEATURES: tuple[str, ...] = (
    "frames_to_ball_gap",
    "control_diff_roll50",
    "control_diff_roll100",
    "near1_diff_leadlag_10f",
    "near1_diff_leadlag_25f",
    "pitch_ctrl_leadlag_10f",
    "pitch_ctrl_leadlag_25f",
)

# Transparent candidate dependencies to inspect during the audit. These are
# not automatic drop instructions.
POTENTIAL_REDUNDANCIES: dict[str, tuple[str, ...]] = {
    "ball_speed_vs_velocity_components": (
        "ball_speed",
        "ball_vx",
        "ball_vy",
    ),
    "ball_status_duplicate_encoding": (
        "ball_status",
        "ball_status_code",
    ),
    "nearest_distance_differences": (
        "home_near1_dist_m",
        "away_near1_dist_m",
        "near1_dist_diff_m",
    ),
    "tti_difference": (
        "home_min_tti_s",
        "away_min_tti_s",
        "tti_diff_s",
    ),
    "nested_player_radius_counts": (
        "home_in_3m",
        "home_in_5m",
        "away_in_3m",
        "away_in_5m",
    ),
    "pitch_control_and_leadlag": (
        "home_pitch_control",
        "pitch_ctrl_leadlag_10f",
        "pitch_ctrl_leadlag_25f",
    ),
    "central_and_causal_control_windows": (
        "control_diff_roll50",
        "control_diff_roll100",
        "control_diff_causal50",
        "control_diff_causal100",
    ),
    "broadcast_camera_attention": (
        "brd_lookat_x",
        "brd_lookat_y",
        "brd_lookat_speed_ms",
        "ball_to_brd_lookat_dist_m",
        "brd_ball_x",
        "brd_ball_y",
    ),
}


# Candidate signed features. Positive values are oriented toward Home, except
# for quantities explicitly named as a non-possession contrast. The original
# source columns are preserved.
SIGNED_FEATURE_FORMULAS: dict[str, str] = {
    "players_in_3m_adv": "home_in_3m - away_in_3m",
    "players_in_5m_adv": "home_in_5m - away_in_5m",
    "near2_dist_adv_m": "away_near2_dist_m - home_near2_dist_m",
    "near3_dist_adv_m": "away_near3_dist_m - home_near3_dist_m",
    "near1_speed_adv_ms": "home_near1_speed_ms - away_near1_speed_ms",
    "near2_speed_adv_ms": "home_near2_speed_ms - away_near2_speed_ms",
    "near3_speed_adv_ms": "home_near3_speed_ms - away_near3_speed_ms",
    "near1_closing_speed_adv_ms": (
        "home_near1_closing_speed_ms - away_near1_closing_speed_ms"
    ),
    "near1_vel_cos_adv": "home_near1_vel_cos - away_near1_vel_cos",
    "consecutive_touches_adv": "home_consecutive_touches - away_consecutive_touches",
    "mean_speed_adv_ms": "home_mean_speed - away_mean_speed",
    "gk_distance_contrast_m": "away_gk_dist_m - home_gk_dist_m",
}


@dataclass(frozen=True)
class SchemaReport:
    """Result of checking an input table against the agreed feature schema."""

    missing_columns: tuple[str, ...]
    unexpected_columns: tuple[str, ...]
    duplicate_columns: tuple[str, ...]
    allow_extra_columns: bool

    @property
    def is_valid(self) -> bool:
        no_disallowed_extras = self.allow_extra_columns or not self.unexpected_columns
        return not self.missing_columns and not self.duplicate_columns and no_disallowed_extras


def expected_input_columns() -> tuple[str, ...]:
    """Return the expected input columns in the declared group order."""
    return tuple(column for group in FEATURE_GROUPS.values() for column in group)


def validate_schema(
    columns: Iterable[str],
    *,
    allow_extra_columns: bool = True,
) -> SchemaReport:
    """Compare actual columns with the agreed schema without changing data."""
    actual = list(columns)
    expected = expected_input_columns()
    expected_set = set(expected)
    actual_set = set(actual)

    seen: set[str] = set()
    duplicates: list[str] = []
    for column in actual:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)

    unexpected = tuple(sorted(actual_set - expected_set)) if allow_extra_columns else tuple(
        sorted(actual_set - expected_set)
    )
    return SchemaReport(
        missing_columns=tuple(column for column in expected if column not in actual_set),
        unexpected_columns=unexpected,
        duplicate_columns=tuple(duplicates),
        allow_extra_columns=allow_extra_columns,
    )


def validate_feature_group_registry() -> None:
    """Raise if the registry itself contains duplicates or misses a column."""
    expected = expected_input_columns()
    if len(expected) != len(set(expected)):
        duplicates = sorted({name for name in expected if expected.count(name) > 1})
        raise ValueError(f"Columns assigned to more than one feature group: {duplicates}")
    if len(expected) != 111:
        raise ValueError(f"Expected 111 registered columns, found {len(expected)}")


def add_candidate_signed_features(
    frame: pd.DataFrame,
    *,
    include_gk_distance_contrast: bool = False,
) -> pd.DataFrame:
    """Add transparent Home-oriented contrasts while preserving source columns.

    The goalkeeper distance contrast is opt-in because distance to a goalkeeper
    is contextual and does not have a universally monotonic relationship with
    team possession.
    """
    out = frame.copy()
    formulas = dict(SIGNED_FEATURE_FORMULAS)
    if not include_gk_distance_contrast:
        formulas.pop("gk_distance_contrast_m")

    for output_name, formula in formulas.items():
        if output_name in out.columns:
            raise ValueError(
                f"Refusing to overwrite existing column {output_name!r}; "
                "choose a new output name or inspect the input schema."
            )
        left_name, operator, right_name = formula.split()
        if left_name not in out.columns or right_name not in out.columns:
            raise KeyError(f"Cannot derive {output_name}: missing {left_name} or {right_name}")
        if operator == "-":
            out[output_name] = out[left_name] - out[right_name]
        else:  # Defensive guard in case the registry is edited later.
            raise ValueError(f"Unsupported operation in formula: {formula}")

    # A centered version keeps the original 0..1 pitch-control probability too.
    if "home_pitch_control" in out.columns:
        if "pitch_control_adv_centered" in out.columns:
            raise ValueError("Refusing to overwrite 'pitch_control_adv_centered'")
        out["pitch_control_adv_centered"] = 2.0 * out["home_pitch_control"] - 1.0

    return out


def build_feature_frame(
    frame: pd.DataFrame,
    *,
    add_signed_features: bool = False,
    include_gk_distance_contrast: bool = False,
    allow_extra_columns: bool = True,
) -> pd.DataFrame:
    """Validate and prepare a feature table; source data remains unmodified.

    Derived candidate features are disabled by default so the first modelling
    run can compare the raw agreed schema against selected transformations.
    """
    report = validate_schema(frame.columns, allow_extra_columns=allow_extra_columns)
    if report.missing_columns:
        raise ValueError(f"Input is missing expected columns: {report.missing_columns}")
    if report.duplicate_columns:
        raise ValueError(f"Input contains duplicate column names: {report.duplicate_columns}")
    if report.unexpected_columns and not allow_extra_columns:
        raise ValueError(f"Input contains unexpected columns: {report.unexpected_columns}")

    out = frame.copy()
    if add_signed_features:
        out = add_candidate_signed_features(
            out,
            include_gk_distance_contrast=include_gk_distance_contrast,
        )
    return out


# Validate the static registry immediately when this module is imported.
validate_feature_group_registry()
