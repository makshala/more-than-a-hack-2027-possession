#!/usr/bin/env python3
"""Convert SkillCorner Open Data into a frame-level H/A training table.

VERSION 3
---------
Phases of Play is the dominant target source and overwrites labels from all
other sources. Tracking ``possession.group`` provides the initial labels;
Dynamic Events fills only frames that are still unlabelled.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs


# ---------------------------------------------------------------------------
# Frozen physical constants
# ---------------------------------------------------------------------------

SKILLCORNER_FPS = 10.0
SKILLCORNER_DT = 1.0 / SKILLCORNER_FPS

PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0

REACTION_TIME_S = 0.40
PLAYER_MAX_SPEED_MS = 7.50
PITCH_CONTROL_TAU_S = 0.50

CONTESTED_RADIUS_M = 2.5
CONTROL_RADIUS_3M = 3.0
SUPPORT_RADIUS_5M = 5.0

SAVGOL_WINDOW = 7
SAVGOL_POLY = 2
BALL_SAVGOL_WINDOW = 5
BALL_SAVGOL_POLY = 2

PLAYER_MAX_KINEMATIC_SPEED_MS = 12.5
BALL_MAX_SPEED_MS = 38.0


# ---------------------------------------------------------------------------
# Model feature contract
# ---------------------------------------------------------------------------

MODEL_FEATURES = [
    "ball_speed",
    "home_near1_dist_m",
    "away_near1_dist_m",
    "home_near2_dist_m",
    "away_near2_dist_m",
    "home_near3_dist_m",
    "away_near3_dist_m",
    "home_in_3m",
    "away_in_3m",
    "home_in_5m",
    "away_in_5m",
    "home_near1_closing_speed_ms",
    "away_near1_closing_speed_ms",
    "home_near1_vel_cos",
    "away_near1_vel_cos",
    "home_min_tti_s",
    "away_min_tti_s",
    "tti_diff_s",
    "home_pitch_control",
    "is_contested",
    "home_centroid_vx",
    "home_centroid_vy",
    "away_centroid_vx",
    "away_centroid_vy",
    "home_length_x",
    "away_length_x",
    "home_width_y",
    "away_width_y",
    "ball_x_norm",
    "ball_vx_norm",
    "home_centroid_x_norm",
    "away_centroid_x_norm",
    "collective_vx_norm",
    "players_in_penalty_box",
    "home_gk_dist_m",
    "away_gk_dist_m",
    "gk_holding_ball_proxy",
]

# Stable CSV report schema. Column order is part of the reporting contract.
REPORT_COLUMNS = [
    "match_id",
    "match_name",
    "tracking_file",
    "metadata_file",
    "json_frames_read",
    "unique_frame_ids",
    "duplicate_frame_ids",
    "out_of_range_frame_ids",
    "tracking_group_home_frames",
    "tracking_group_away_frames",
    "dynamic_pp_home_frames",
    "dynamic_pp_away_frames",
    "phase_fallback_home_frames",
    "phase_fallback_away_frames",
    "labelled_home_frames",
    "labelled_away_frames",
    "unlabelled_or_unknown_possession_frames",
    "raw_tracking_unknown_possession_frames",
    "labelled_frames_without_ball_speed",
    "frames_without_ball_xy_in_tracking",
    "unknown_player_ids_not_in_match_metadata",
    "unknown_player_id_examples",
    "unmapped_possession_player_ids",
    "possession_player_team_mismatch_frames",
    "home_roster_players",
    "away_roster_players",
    "label_coverage_pct_of_json_frames",
    "ball_speed_valid_pct_of_labelled",
    "dynamic_pp_file_found",
    "dynamic_pp_rows_considered",
    "dynamic_pp_frames_filled",
    "dynamic_pp_conflict_frames",
    "dynamic_pp_unmapped_team_rows",
    "dynamic_pp_schema_ok",
    "phases_file_found",
    "phases_rows_considered",
    "phases_frames_filled",
    "phases_conflict_frames",
    "phases_unmapped_team_rows",
    "phases_schema_ok",
    *[f"nan_pct__{feature}" for feature in MODEL_FEATURES],
]


# ---------------------------------------------------------------------------
# Value parsing and metadata helpers
# ---------------------------------------------------------------------------

def _id_key(value: Any) -> str:
    """Convert IDs to a consistent string representation."""
    if value is None:
        return ""

    if (
        isinstance(value, (float, np.floating))
        and np.isfinite(value)
        and float(value).is_integer()
    ):
        return str(int(value))

    text = str(value).strip()

    try:
        numeric = float(text)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass

    return text


def _as_float(value: Any) -> float:
    """Convert a value to a finite float; return NaN when conversion fails."""
    try:
        if value is None:
            return np.nan

        out = float(value)
        return out if math.isfinite(out) else np.nan

    except (TypeError, ValueError):
        return np.nan


def _as_timestamp_seconds(value: Any) -> float:
    """Convert a numeric or colon-separated timestamp to seconds."""
    numeric = _as_float(value)
    if np.isfinite(numeric):
        return numeric

    if isinstance(value, str) and ":" in value:
        try:
            parts = value.strip().split(":")

            if len(parts) == 3:
                return (
                    float(parts[0]) * 3600.0
                    + float(parts[1]) * 60.0
                    + float(parts[2])
                )

            if len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])

        except ValueError:
            pass

    return np.nan


def _normalise_possession_group(possession: Any) -> int:
    """Map tracking possession groups to H/A targets (0/1), or -1."""
    if not isinstance(possession, dict):
        return -1

    group = (
        str(possession.get("group", ""))
        .strip()
        .lower()
        .replace("_", " ")
    )

    if group in {"home team", "home"}:
        return 0

    if group in {"away team", "away"}:
        return 1

    return -1


def _possession_player_id(possession: Any) -> str:
    """Extract an optional player ID from a tracking possession object."""
    if not isinstance(possession, dict):
        return ""

    player_id = possession.get("player_id")

    if player_id is None:
        player = possession.get("player")
        if isinstance(player, dict):
            player_id = player.get("id", player.get("player_id"))
        elif player is not None:
            player_id = player

    if isinstance(player_id, dict):
        player_id = player_id.get("id", player_id.get("player_id"))

    return _id_key(player_id)


def _load_match_metadata(
    path: Path,
) -> tuple[dict[str, int], dict[str, int], dict[str, Any], int]:
    """Load team/player mappings and derive the total frame count."""
    with path.open("r", encoding="utf-8") as file:
        meta = json.load(file)

    home_team_id = _id_key(meta.get("home_team", {}).get("id"))
    away_team_id = _id_key(meta.get("away_team", {}).get("id"))

    team_to_target = {
        home_team_id: 0,
        away_team_id: 1,
    }

    player_team: dict[str, int] = {}

    for player in meta.get("players", []):
        player_id = _id_key(player.get("id"))
        team_id = _id_key(player.get("team_id"))

        if player_id and team_id in team_to_target:
            player_team[player_id] = team_to_target[team_id]

    periods = meta.get("match_periods", [])
    max_frame = max(
        (int(period.get("end_frame", 0) or 0) for period in periods),
        default=0,
    )

    return player_team, team_to_target, meta, max_frame + 1


def _period_segments(period: np.ndarray) -> list[tuple[int, int, int]]:
    """Return contiguous frame ranges for periods 1 and 2."""
    segments: list[tuple[int, int, int]] = []

    for period_number in (1, 2):
        indices = np.flatnonzero(period == period_number)

        if indices.size == 0:
            continue

        starts = [int(indices[0])]
        ends: list[int] = []

        for index in range(1, len(indices)):
            if indices[index] != indices[index - 1] + 1:
                ends.append(int(indices[index - 1]) + 1)
                starts.append(int(indices[index]))

        ends.append(int(indices[-1]) + 1)

        segments.extend(
            (period_number, start, end)
            for start, end in zip(starts, ends)
        )

    return segments


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def _player_velocity_like_team_pipeline(
    px: np.ndarray,
    py: np.ndarray,
    period: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate player velocities using the original team-pipeline method."""
    n_frames, n_players = px.shape

    vx = np.zeros((n_frames, n_players), dtype=np.float32)
    vy = np.zeros((n_frames, n_players), dtype=np.float32)

    if n_players == 0:
        return vx, vy

    c_pos = savgol_coeffs(
        SAVGOL_WINDOW,
        SAVGOL_POLY,
        deriv=0,
        use="conv",
    ).astype(np.float32)

    c_vel = (
        savgol_coeffs(
            SAVGOL_WINDOW,
            SAVGOL_POLY,
            deriv=1,
            use="conv",
        )
        / dt
    ).astype(np.float32)

    kernel = np.ones(SAVGOL_WINDOW, dtype=np.float32)

    for _, start, end in _period_segments(period):
        x = px[start:end]
        y = py[start:end]

        valid = np.isfinite(x) & np.isfinite(y)

        x0 = np.where(valid, x, 0.0).astype(np.float32)
        y0 = np.where(valid, y, 0.0).astype(np.float32)

        has_full_window = (
            convolve1d(
                valid.astype(np.float32),
                kernel,
                axis=0,
                mode="constant",
                cval=0.0,
            )
            > (SAVGOL_WINDOW - 0.5)
        )

        xs = convolve1d(
            x0,
            c_pos,
            axis=0,
            mode="nearest",
        )
        ys = convolve1d(
            y0,
            c_pos,
            axis=0,
            mode="nearest",
        )

        x_smooth = np.where(has_full_window, xs, x).astype(np.float32)
        y_smooth = np.where(has_full_window, ys, y).astype(np.float32)

        vx_sg = convolve1d(
            x0,
            c_vel,
            axis=0,
            mode="nearest",
        )
        vy_sg = convolve1d(
            y0,
            c_vel,
            axis=0,
            mode="nearest",
        )

        dx_f = np.diff(
            x_smooth,
            axis=0,
            append=np.full((1, n_players), np.nan),
        )
        dy_f = np.diff(
            y_smooth,
            axis=0,
            append=np.full((1, n_players), np.nan),
        )
        dx_b = np.diff(
            x_smooth,
            axis=0,
            prepend=np.full((1, n_players), np.nan),
        )
        dy_b = np.diff(
            y_smooth,
            axis=0,
            prepend=np.full((1, n_players), np.nan),
        )

        has_f = np.isfinite(dx_f)
        has_b = np.isfinite(dx_b)

        vx_fb = np.where(
            has_f & has_b,
            0.5 * (dx_f + dx_b) / dt,
            np.where(
                has_f,
                dx_f / dt,
                np.where(has_b, dx_b / dt, 0.0),
            ),
        )
        vy_fb = np.where(
            has_f & has_b,
            0.5 * (dy_f + dy_b) / dt,
            np.where(
                has_f,
                dy_f / dt,
                np.where(has_b, dy_b / dt, 0.0),
            ),
        )

        vx_seg = np.where(
            has_full_window,
            vx_sg,
            vx_fb,
        ).astype(np.float32)
        vy_seg = np.where(
            has_full_window,
            vy_sg,
            vy_fb,
        ).astype(np.float32)

        speed = np.hypot(vx_seg, vy_seg)
        invalid = (~valid) | (speed > PLAYER_MAX_KINEMATIC_SPEED_MS)

        vx_seg[invalid] = 0.0
        vy_seg[invalid] = 0.0

        vx[start:end] = vx_seg
        vy[start:end] = vy_seg

    return vx, vy


def _ball_kinematics_like_team_pipeline(
    bx: np.ndarray,
    by: np.ndarray,
    ball_detected: np.ndarray,
    period: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate ball velocity and speed using the original pipeline method."""
    n_frames = len(bx)

    vx = np.full(n_frames, np.nan, dtype=np.float32)
    vy = np.full(n_frames, np.nan, dtype=np.float32)
    speed = np.full(n_frames, np.nan, dtype=np.float32)

    c_vel = (
        savgol_coeffs(
            BALL_SAVGOL_WINDOW,
            BALL_SAVGOL_POLY,
            deriv=1,
            use="conv",
        )
        / dt
    ).astype(np.float32)

    kernel = np.ones(BALL_SAVGOL_WINDOW, dtype=np.float32)

    for _, start, end in _period_segments(period):
        x = bx[start:end].astype(np.float32, copy=True)
        y = by[start:end].astype(np.float32, copy=True)

        valid_xy = np.isfinite(x) & np.isfinite(y)

        if valid_xy.sum() < 2:
            continue

        indices = np.arange(len(x))

        x_fill = np.interp(
            indices,
            indices[valid_xy],
            x[valid_xy],
        ).astype(np.float32)
        y_fill = np.interp(
            indices,
            indices[valid_xy],
            y[valid_xy],
        ).astype(np.float32)

        status = np.where(
            valid_xy,
            np.where(ball_detected[start:end] == 1, 0, 1),
            2,
        ).astype(np.int8)

        is_obs = (status <= 1).astype(np.float32)

        has_5 = (
            (
                convolve1d(
                    is_obs,
                    kernel,
                    mode="constant",
                    cval=0.0,
                )
                > 4.5
            )
            | (
                convolve1d(
                    is_obs,
                    kernel,
                    mode="constant",
                    cval=0.0,
                )
                < 0.5
            )
        )

        vx_sg = convolve1d(x_fill, c_vel, mode="nearest")
        vy_sg = convolve1d(y_fill, c_vel, mode="nearest")

        same_next = (status[:-1] <= 1) == (status[1:] <= 1)
        has_f = np.concatenate((same_next, [False]))
        has_b = np.concatenate(([False], same_next))

        dx_f = np.diff(x_fill, append=x_fill[-1])
        dy_f = np.diff(y_fill, append=y_fill[-1])
        dx_b = np.diff(x_fill, prepend=x_fill[0])
        dy_b = np.diff(y_fill, prepend=y_fill[0])

        vx_fb = np.where(
            has_f & has_b,
            0.5 * (dx_f + dx_b) / dt,
            np.where(
                has_f,
                dx_f / dt,
                np.where(has_b, dx_b / dt, 0.0),
            ),
        )
        vy_fb = np.where(
            has_f & has_b,
            0.5 * (dy_f + dy_b) / dt,
            np.where(
                has_f,
                dy_f / dt,
                np.where(has_b, dy_b / dt, 0.0),
            ),
        )

        vx_seg = np.where(has_5, vx_sg, vx_fb).astype(np.float32)
        vy_seg = np.where(has_5, vy_sg, vy_fb).astype(np.float32)

        speed_seg = np.hypot(vx_seg, vy_seg)
        over = speed_seg > BALL_MAX_SPEED_MS
        scale = BALL_MAX_SPEED_MS / np.maximum(speed_seg, 1e-5)

        vx_seg = np.where(
            over,
            vx_seg * scale,
            vx_seg,
        ).astype(np.float32)
        vy_seg = np.where(
            over,
            vy_seg * scale,
            vy_seg,
        ).astype(np.float32)

        speed_seg = np.minimum(
            speed_seg,
            BALL_MAX_SPEED_MS,
        ).astype(np.float32)

        vx_seg[~valid_xy] = np.nan
        vy_seg[~valid_xy] = np.nan
        speed_seg[~valid_xy] = np.nan

        vx[start:end] = vx_seg
        vy[start:end] = vy_seg
        speed[start:end] = speed_seg

    return vx, vy, speed


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_features(
    px: np.ndarray,
    py: np.ndarray,
    detected: np.ndarray,
    player_team: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    period: np.ndarray,
    ball_detected: np.ndarray,
) -> pd.DataFrame:
    """Build the frame-level feature table using the original definitions."""
    n_frames = len(period)

    pvx, pvy = _player_velocity_like_team_pipeline(
        px,
        py,
        period,
        SKILLCORNER_DT,
    )
    ball_vx, ball_vy, ball_speed = _ball_kinematics_like_team_pipeline(
        bx,
        by,
        ball_detected,
        period,
        SKILLCORNER_DT,
    )

    out: dict[str, np.ndarray] = {
        "ball_speed": ball_speed,
    }

    team_cx: dict[str, np.ndarray] = {}
    ball_xy_valid = np.isfinite(bx) & np.isfinite(by)

    for code, team in ((0, "home"), (1, "away")):
        mask = player_team == code

        tx = px[:, mask]
        ty = py[:, mask]
        tvx = pvx[:, mask]
        tvy = pvy[:, mask]

        valid = np.isfinite(tx) & np.isfinite(ty)
        n_players = tx.shape[1]

        n_valid = valid.sum(axis=1)
        count = np.maximum(1, n_valid).astype(np.float32)

        cx = (
            np.where(valid, tx, 0.0).sum(axis=1) / count
        )
        cy = (
            np.where(valid, ty, 0.0).sum(axis=1) / count
        )

        cx[n_valid == 0] = 0.0
        cy[n_valid == 0] = 0.0

        team_cx[team] = cx.astype(np.float32)

        c_vx = np.zeros(n_frames, dtype=np.float32)
        c_vy = np.zeros(n_frames, dtype=np.float32)

        for _, start, end in _period_segments(period):
            if end - start >= 2:
                c_vx[start:end] = np.gradient(
                    cx[start:end],
                    SKILLCORNER_DT,
                )
                c_vy[start:end] = np.gradient(
                    cy[start:end],
                    SKILLCORNER_DT,
                )

        out[f"{team}_centroid_vx"] = c_vx
        out[f"{team}_centroid_vy"] = c_vy

        var_x = (
            np.where(valid, (tx - cx[:, None]) ** 2, 0.0).sum(axis=1)
            / count
        )
        var_y = (
            np.where(valid, (ty - cy[:, None]) ** 2, 0.0).sum(axis=1)
            / count
        )

        out[f"{team}_length_x"] = (
            2.0 * np.sqrt(np.maximum(0.0, var_x))
        ).astype(np.float32)
        out[f"{team}_width_y"] = (
            2.0 * np.sqrt(np.maximum(0.0, var_y))
        ).astype(np.float32)

        dx = bx[:, None] - tx
        dy = by[:, None] - ty
        dist = np.hypot(dx, dy)

        dist[~valid | ~ball_xy_valid[:, None]] = np.inf

        order = np.argsort(dist, axis=1)[:, :3]
        sorted_dist = np.take_along_axis(dist, order, axis=1)
        sorted_vx = np.take_along_axis(tvx, order, axis=1)
        sorted_vy = np.take_along_axis(tvy, order, axis=1)

        sorted_dist[~np.isfinite(sorted_dist)] = np.nan

        for k in range(3):
            out[f"{team}_near{k + 1}_dist_m"] = (
                sorted_dist[:, k].astype(np.float32)
            )

        for radius, suffix in (
            (CONTROL_RADIUS_3M, "3m"),
            (SUPPORT_RADIUS_5M, "5m"),
        ):
            counts = np.sum(
                valid & (dist <= radius),
                axis=1,
            ).astype(np.float32)

            out[f"{team}_in_{suffix}"] = np.where(
                ball_xy_valid,
                counts,
                np.nan,
            )

        near1_dist = sorted_dist[:, 0]
        n1_vx = sorted_vx[:, 0]
        n1_vy = sorted_vy[:, 0]

        n1_dx = np.take_along_axis(
            dx,
            order[:, :1],
            axis=1,
        )[:, 0]
        n1_dy = np.take_along_axis(
            dy,
            order[:, :1],
            axis=1,
        )[:, 0]

        n1_valid = (
            np.isfinite(near1_dist)
            & np.isfinite(bx)
            & np.isfinite(by)
        )

        out[f"_{team}_near1_vx"] = np.where(
            n1_valid,
            n1_vx,
            np.nan,
        )
        out[f"_{team}_near1_vy"] = np.where(
            n1_valid,
            n1_vy,
            np.nan,
        )
        out[f"_{team}_near1_dx"] = np.where(
            n1_valid,
            n1_dx,
            np.nan,
        )
        out[f"_{team}_near1_dy"] = np.where(
            n1_valid,
            n1_dy,
            np.nan,
        )
        out[f"_{team}_near1_dist_tmp"] = np.where(
            n1_valid,
            near1_dist,
            np.nan,
        )

        proj_x = tx + tvx * REACTION_TIME_S
        proj_y = ty + tvy * REACTION_TIME_S

        tti = (
            REACTION_TIME_S
            + np.hypot(
                bx[:, None] - proj_x,
                by[:, None] - proj_y,
            )
            / PLAYER_MAX_SPEED_MS
        )
        tti[~valid] = 99.0

        min_tti = np.minimum(
            np.min(tti, axis=1),
            20.0,
        ).astype(np.float32)

        out[f"{team}_min_tti_s"] = np.where(
            ball_xy_valid,
            min_tti,
            np.nan,
        )

        rates = np.where(
            valid & ball_xy_valid[:, None],
            np.exp(-tti / PITCH_CONTROL_TAU_S),
            0.0,
        ).sum(axis=1)

        out[f"_{team}_pitch_control_rate"] = np.where(
            ball_xy_valid,
            rates,
            np.nan,
        ).astype(np.float32)

    for team in ("home", "away"):
        n1_vx = out[f"_{team}_near1_vx"]
        n1_vy = out[f"_{team}_near1_vy"]
        n1_dx = out[f"_{team}_near1_dx"]
        n1_dy = out[f"_{team}_near1_dy"]
        n1_dist = out[f"_{team}_near1_dist_tmp"]

        close = (
            (n1_vx - ball_vx) * n1_dx
            + (n1_vy - ball_vy) * n1_dy
        ) / (n1_dist + 0.1)

        out[f"{team}_near1_closing_speed_ms"] = np.clip(
            close,
            -25.0,
            25.0,
        ).astype(np.float32)

        player_speed = np.hypot(n1_vx, n1_vy)
        cos = (
            n1_vx * ball_vx + n1_vy * ball_vy
        ) / (player_speed * ball_speed + 1e-5)

        cos_valid = (player_speed > 0.3) & (ball_speed > 0.3)

        out[f"{team}_near1_vel_cos"] = np.where(
            np.isfinite(bx) & np.isfinite(by),
            np.where(
                cos_valid,
                np.clip(cos, -1.0, 1.0),
                0.0,
            ),
            np.nan,
        ).astype(np.float32)

    out["near1_dist_diff_m"] = (
        out["away_near1_dist_m"] - out["home_near1_dist_m"]
    ).astype(np.float32)

    out["tti_diff_s"] = (
        out["away_min_tti_s"] - out["home_min_tti_s"]
    ).astype(np.float32)

    total_rate = (
        out["_home_pitch_control_rate"]
        + out["_away_pitch_control_rate"]
    )

    pc = np.where(
        total_rate > 1e-6,
        out["_home_pitch_control_rate"] / total_rate,
        0.5,
    )

    out["home_pitch_control"] = np.where(
        ball_xy_valid,
        pc,
        np.nan,
    ).astype(np.float32)

    contested = (
        (out["home_near1_dist_m"] <= CONTESTED_RADIUS_M)
        & (out["away_near1_dist_m"] <= CONTESTED_RADIUS_M)
    )

    out["is_contested"] = np.where(
        ball_xy_valid,
        contested.astype(np.float32),
        np.nan,
    ).astype(np.float32)

    flip = np.zeros(n_frames, dtype=bool)

    for period_number in (1, 2):
        period_mask = period == period_number

        if period_mask.any():
            flip[period_mask] = (
                float(np.mean(team_cx["home"][period_mask]))
                > float(np.mean(team_cx["away"][period_mask]))
            )

    bx_origin = bx + PITCH_LENGTH_M / 2.0
    by_origin = by + PITCH_WIDTH_M / 2.0

    out["ball_x_norm"] = np.where(
        flip,
        PITCH_LENGTH_M - bx_origin,
        bx_origin,
    ).astype(np.float32)

    out["ball_vx_norm"] = np.where(
        flip,
        -ball_vx,
        ball_vx,
    ).astype(np.float32)

    out["home_centroid_x_norm"] = np.where(
        flip,
        PITCH_LENGTH_M - (
            team_cx["home"] + PITCH_LENGTH_M / 2.0
        ),
        team_cx["home"] + PITCH_LENGTH_M / 2.0,
    ).astype(np.float32)

    out["away_centroid_x_norm"] = np.where(
        flip,
        PITCH_LENGTH_M - (
            team_cx["away"] + PITCH_LENGTH_M / 2.0
        ),
        team_cx["away"] + PITCH_LENGTH_M / 2.0,
    ).astype(np.float32)

    out["collective_vx_norm"] = (
        0.5
        * (
            np.where(
                flip,
                -out["home_centroid_vx"],
                out["home_centroid_vx"],
            )
            + np.where(
                flip,
                -out["away_centroid_vx"],
                out["away_centroid_vx"],
            )
        )
    ).astype(np.float32)

    px_origin = px + PITCH_LENGTH_M / 2.0
    py_origin = py + PITCH_WIDTH_M / 2.0

    player_valid = np.isfinite(px) & np.isfinite(py)

    in_left = (
        player_valid
        & (px_origin <= 16.5)
        & (py_origin >= 13.84)
        & (py_origin <= 54.16)
    )
    in_right = (
        player_valid
        & (px_origin >= PITCH_LENGTH_M - 16.5)
        & (py_origin >= 13.84)
        & (py_origin <= 54.16)
    )

    out["players_in_penalty_box"] = np.maximum(
        in_left.sum(axis=1),
        in_right.sum(axis=1),
    ).astype(np.int8)

    px_norm = np.where(
        flip[:, None],
        PITCH_LENGTH_M - px_origin,
        px_origin,
    )

    home_mask = player_team == 0
    away_mask = player_team == 1

    hx = px_norm[:, home_mask]
    ax = px_norm[:, away_mask]
    hy = py_origin[:, home_mask]
    ay = py_origin[:, away_mask]

    hvalid = np.isfinite(hx) & np.isfinite(hy)
    avalid = np.isfinite(ax) & np.isfinite(ay)

    h_idx = np.argmin(
        np.where(hvalid, hx, 999.0),
        axis=1,
    )
    a_idx = np.argmax(
        np.where(avalid, ax, -999.0),
        axis=1,
    )

    rows = np.arange(n_frames)

    h_gx = hx[rows, h_idx]
    h_gy = hy[rows, h_idx]
    a_gx = ax[rows, a_idx]
    a_gy = ay[rows, a_idx]

    h_ok = hvalid[rows, h_idx]
    a_ok = avalid[rows, a_idx]

    h_dist = np.where(
        h_ok & ball_xy_valid,
        np.hypot(
            out["ball_x_norm"] - h_gx,
            by_origin - h_gy,
        ),
        120.0,
    )
    a_dist = np.where(
        a_ok & ball_xy_valid,
        np.hypot(
            out["ball_x_norm"] - a_gx,
            by_origin - a_gy,
        ),
        120.0,
    )

    out["home_gk_dist_m"] = np.minimum(
        h_dist,
        120.0,
    ).astype(np.float32)

    out["away_gk_dist_m"] = np.minimum(
        a_dist,
        120.0,
    ).astype(np.float32)

    h_hold = (
        (out["home_gk_dist_m"] <= 3.5)
        & (out["ball_x_norm"] <= 20.0)
        & (out["away_near1_dist_m"] > 4.5)
    )
    a_hold = (
        (out["away_gk_dist_m"] <= 3.5)
        & (out["ball_x_norm"] >= PITCH_LENGTH_M - 20.0)
        & (out["home_near1_dist_m"] > 4.5)
    )

    out["gk_holding_ball_proxy"] = np.where(
        h_hold,
        1,
        np.where(a_hold, -1, 0),
    ).astype(np.int8)

    for key in list(out):
        if key.startswith("_"):
            del out[key]

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Interval-based labels
# ---------------------------------------------------------------------------

def _apply_interval_labels(
    csv_path: Path,
    target: np.ndarray,
    label_source: np.ndarray,
    period: np.ndarray,
    team_to_target: dict[str, int],
    source_code: int,
    team_column: str,
    source_name: str,
    event_type_filter: str | None = None,
    overwrite: bool = False,
    end_exclusive: bool = False,
) -> dict[str, Any]:
    """Apply interval labels and collect source-specific diagnostics.

    When ``overwrite`` is True (Phases of Play), labels replace existing labels
    in every covered frame. When it is False (Dynamic Events), only currently
    unlabelled frames are filled. Diagnostic counters do not affect labels.
    """
    stats: dict[str, Any] = {
        f"{source_name}_file_found": bool(csv_path.exists()),
        f"{source_name}_schema_ok": False,
        f"{source_name}_rows_considered": 0,
        f"{source_name}_frames_filled": 0,
        f"{source_name}_conflict_frames": 0,
        f"{source_name}_unmapped_team_rows": 0,
    }

    if not csv_path.exists():
        return stats

    df = pd.read_csv(csv_path)

    required = {
        "frame_start",
        "frame_end",
        "period",
        team_column,
    }

    if not required.issubset(df.columns):
        return stats

    if event_type_filter is not None and "event_type" not in df.columns:
        return stats

    stats[f"{source_name}_schema_ok"] = True

    if event_type_filter is not None:
        df = df[
            df["event_type"].astype(str).str.strip().str.lower()
            == event_type_filter
        ]

    n_frames = len(target)

    for row in df.itertuples(index=False):
        row_dict = row._asdict()

        team_id = _id_key(row_dict.get(team_column))
        team_label = team_to_target.get(team_id)

        if team_label is None:
            stats[f"{source_name}_unmapped_team_rows"] += 1
            continue

        start_f = _as_float(row_dict.get("frame_start"))
        end_f = _as_float(row_dict.get("frame_end"))

        period_value = _as_float(row_dict.get("period"))
        period_number = int(period_value) if np.isfinite(period_value) else 0

        if not (
            np.isfinite(start_f)
            and np.isfinite(end_f)
            and period_number in (1, 2)
        ):
            continue

        stats[f"{source_name}_rows_considered"] += 1

        start = max(0, int(start_f))
        end_value = int(end_f)

        stop = min(
            n_frames,
            end_value if end_exclusive else end_value + 1,
        )

        if stop <= start:
            continue

        indices = np.arange(
            start,
            stop,
            dtype=np.int32,
        )

        indices = indices[period[indices] == period_number]

        if indices.size == 0:
            continue

        conflicts = (
            (target[indices] >= 0)
            & (target[indices] != team_label)
        )

        stats[f"{source_name}_conflict_frames"] += int(
            conflicts.sum()
        )

        fill_indices = (
            indices
            if overwrite
            else indices[target[indices] < 0]
        )

        if fill_indices.size:
            target[fill_indices] = team_label
            label_source[fill_indices] = source_code

            stats[f"{source_name}_frames_filled"] += int(
                fill_indices.size
            )

    return stats


# ---------------------------------------------------------------------------
# Match conversion
# ---------------------------------------------------------------------------

def convert_match(match_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert one match directory and return its table and CSV report row."""
    match_jsons = sorted(match_dir.glob("*_match.json"))
    tracking_files = sorted(
        match_dir.glob("*_tracking_extrapolated.jsonl")
    )

    match_json = match_jsons[0]
    tracking_path = tracking_files[0]

    match_id = match_json.name.split("_", 1)[0]

    (
        player_team_map,
        team_to_target,
        metadata,
        n_frames,
    ) = _load_match_metadata(match_json)

    home_team_name = metadata.get("home_team", {}).get("name", "")
    away_team_name = metadata.get("away_team", {}).get("name", "")
    match_name = f"{home_team_name} vs {away_team_name}"

    all_player_ids = sorted(player_team_map.keys())
    player_slot = {
        player_id: index
        for index, player_id in enumerate(all_player_ids)
    }

    player_team = np.array(
        [player_team_map[player_id] for player_id in all_player_ids],
        dtype=np.int8,
    )
    n_players = len(all_player_ids)

    px = np.full(
        (n_frames, n_players),
        np.nan,
        dtype=np.float32,
    )
    py = np.full(
        (n_frames, n_players),
        np.nan,
        dtype=np.float32,
    )
    detected = np.zeros(
        (n_frames, n_players),
        dtype=bool,
    )

    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)

    ball_detected = np.zeros(n_frames, dtype=np.int8)
    period = np.zeros(n_frames, dtype=np.int8)
    timestamp = np.full(n_frames, np.nan, dtype=np.float32)

    target = np.full(n_frames, -1, dtype=np.int8)
    label_source = np.zeros(n_frames, dtype=np.int8)

    # Keep raw tracking labels separately for report diagnostics.
    tracking_target = np.full(n_frames, -1, dtype=np.int8)
    seen_frames: set[int] = set()
    duplicate_frame_ids = 0
    out_of_range_frame_ids: set[int] = set()
    json_frames_read = 0

    unknown_player_ids: set[str] = set()
    unmapped_possession_player_ids: set[str] = set()
    possession_team_mismatch_frames: set[int] = set()

    # 1. Tracking possession.group: initial labels (source code 1).
    with tracking_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("version"):
                continue

            try:
             rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                     f"Błąd JSON w pliku {tracking_path.name}. "
                     f"Zawartość problematycznej linii: '{line[:50]}...'. "
                     f"Oryginalny błąd: {exc}"
             ) from exc
            json_frames_read += 1

            frame = int(rec.get("frame", -1))

            if frame < 0 or frame >= n_frames:
                out_of_range_frame_ids.add(frame)
                continue

            if frame in seen_frames:
                duplicate_frame_ids += 1
            else:
                seen_frames.add(frame)

            period_value = int(rec.get("period") or 0)
            period[frame] = (
                period_value
                if period_value in (1, 2)
                else 0
            )
            timestamp[frame] = _as_timestamp_seconds(
                rec.get("timestamp")
            )

            # Label source 1: Tracking possession.group.
            possession = rec.get("possession")
            tracking_target[frame] = _normalise_possession_group(
                possession
            )
            target[frame] = tracking_target[frame]

            if target[frame] >= 0:
                label_source[frame] = 1

            # Possession-player diagnostics are not used for target assignment.
            possession_player_id = _possession_player_id(possession)
            if possession_player_id:
                possession_player_team = player_team_map.get(
                    possession_player_id
                )
                if possession_player_team is None:
                    unmapped_possession_player_ids.add(
                        possession_player_id
                    )
                else:
                    possession_group = tracking_target[frame]
                    if (
                        possession_group >= 0
                        and possession_player_team != possession_group
                    ):
                        possession_team_mismatch_frames.add(frame)

            ball = rec.get("ball_data") or {}

            if isinstance(ball, dict):
                bx[frame] = _as_float(ball.get("x"))
                by[frame] = _as_float(ball.get("y"))
                ball_detected[frame] = int(
                    bool(ball.get("is_detected_ball", False))
                )

            for player in rec.get("player_data") or []:
                if not isinstance(player, dict):
                    continue

                player_id = _id_key(player.get("player_id"))
                slot = player_slot.get(player_id)

                if slot is None:
                    if player_id:
                        unknown_player_ids.add(player_id)
                    continue

                x = _as_float(player.get("x"))
                y = _as_float(player.get("y"))

                if np.isfinite(x) and np.isfinite(y):
                    px[frame, slot] = x
                    py[frame, slot] = y
                    detected[frame, slot] = bool(
                        player.get("is_detected", False)
                    )

    fallback_stats = {}

    # 2. Dynamic Events: fill only unlabelled frames (source code 2).
    fallback_stats.update(
        _apply_interval_labels(
            match_dir / f"{match_id}_dynamic_events.csv",
            target,
            label_source,
            period,
            team_to_target,
            source_code=2,
            team_column="team_id",
            source_name="dynamic_pp",
            event_type_filter="player_possession",
            overwrite=False,
            end_exclusive=False,
        )
    )

    # 3. Phases of Play: dominant source; overwrite existing labels (code 3).
    fallback_stats.update(
        _apply_interval_labels(
            match_dir / f"{match_id}_phases_of_play.csv",
            target,
            label_source,
            period,
            team_to_target,
            source_code=3,
            team_column="team_in_possession_id",
            source_name="phases",
            event_type_filter=None,
            overwrite=True,
            end_exclusive=True,
        )
    )

    feature_df = _compute_features(
        px,
        py,
        detected,
        player_team,
        bx,
        by,
        period,
        ball_detected,
    )

    feature_df["label_source"] = label_source

    feature_df.insert(0, "timestamp_s", timestamp)
    feature_df.insert(0, "period", period)
    feature_df.insert(
        0,
        "frame",
        np.arange(n_frames, dtype=np.int32),
    )
    feature_df.insert(0, "match_id", str(match_id))
    feature_df.insert(0, "target", target)

    labelled = feature_df[
        feature_df["target"].isin([0, 1])
        & feature_df["period"].isin([1, 2])
    ].copy()

    valid_period_mask = np.isin(period, [1, 2])

    labelled_home = int(
        ((target == 0) & valid_period_mask).sum()
    )
    labelled_away = int(
        ((target == 1) & valid_period_mask).sum()
    )

    tracking_home = int(
        (
            (target == 0)
            & (label_source == 1)
            & valid_period_mask
        ).sum()
    )
    tracking_away = int(
        (
            (target == 1)
            & (label_source == 1)
            & valid_period_mask
        ).sum()
    )
    dynamic_home = int(
        (
            (target == 0)
            & (label_source == 2)
            & valid_period_mask
        ).sum()
    )
    dynamic_away = int(
        (
            (target == 1)
            & (label_source == 2)
            & valid_period_mask
        ).sum()
    )
    phases_home = int(
        (
            (target == 0)
            & (label_source == 3)
            & valid_period_mask
        ).sum()
    )
    phases_away = int(
        (
            (target == 1)
            & (label_source == 3)
            & valid_period_mask
        ).sum()
    )

    seen_indices = np.fromiter(
        seen_frames,
        dtype=np.int64,
        count=len(seen_frames),
    )

    raw_tracking_unknown = int(
        (tracking_target[seen_indices] < 0).sum()
    ) if seen_frames else 0

    frames_without_ball_xy = int(
        (
            ~(
                np.isfinite(bx[seen_indices])
                & np.isfinite(by[seen_indices])
            )
        ).sum()
    ) if seen_frames else 0

    labelled_frames_without_ball_speed = int(
        labelled["ball_speed"].isna().sum()
    )

    label_coverage_pct = (
        100.0 * len(labelled) / json_frames_read
        if json_frames_read
        else 0.0
    )
    ball_speed_valid_pct = (
        100.0 * labelled["ball_speed"].notna().sum() / len(labelled)
        if len(labelled)
        else 0.0
    )

    feature_nan_pct = {
        f"nan_pct__{feature}": round(
            float(labelled[feature].isna().mean() * 100.0)
            if len(labelled)
            else 0.0,
            3,
        )
        for feature in MODEL_FEATURES
    }

    report = {
        "match_id": str(match_id),
        "match_name": match_name,
        "tracking_file": tracking_path.name,
        "metadata_file": match_json.name,
        "json_frames_read": int(json_frames_read),
        "unique_frame_ids": int(len(seen_frames)),
        "duplicate_frame_ids": int(duplicate_frame_ids),
        "out_of_range_frame_ids": int(len(out_of_range_frame_ids)),
        "tracking_group_home_frames": tracking_home,
        "tracking_group_away_frames": tracking_away,
        "dynamic_pp_home_frames": dynamic_home,
        "dynamic_pp_away_frames": dynamic_away,
        "phase_fallback_home_frames": phases_home,
        "phase_fallback_away_frames": phases_away,
        "labelled_home_frames": labelled_home,
        "labelled_away_frames": labelled_away,
        "unlabelled_or_unknown_possession_frames": int(
            ((target < 0) & valid_period_mask).sum()
        ),
        "raw_tracking_unknown_possession_frames": raw_tracking_unknown,
        "labelled_frames_without_ball_speed": (
            labelled_frames_without_ball_speed
        ),
        "frames_without_ball_xy_in_tracking": frames_without_ball_xy,
        "unknown_player_ids_not_in_match_metadata": len(
            unknown_player_ids
        ),
        "unknown_player_id_examples": ";".join(
            sorted(unknown_player_ids)[:10]
        ),
        "unmapped_possession_player_ids": len(
            unmapped_possession_player_ids
        ),
        "possession_player_team_mismatch_frames": len(
            possession_team_mismatch_frames
        ),
        "home_roster_players": int(
            sum(team == 0 for team in player_team_map.values())
        ),
        "away_roster_players": int(
            sum(team == 1 for team in player_team_map.values())
        ),
        "label_coverage_pct_of_json_frames": round(
            label_coverage_pct,
            2,
        ),
        "ball_speed_valid_pct_of_labelled": round(
            ball_speed_valid_pct,
            2,
        ),
        **fallback_stats,
        **feature_nan_pct,
    }

    return labelled, report


# ---------------------------------------------------------------------------
# Dataset discovery and command-line interface
# ---------------------------------------------------------------------------

def discover_match_dirs(
    data_root: Path,
    match_ids: Iterable[str] | None,
) -> list[Path]:
    """Return match directories selected by ID or discovered in data_root."""
    if match_ids:
        dirs = [data_root / str(match_id) for match_id in match_ids]
    else:
        dirs = sorted(
            path
            for path in data_root.iterdir()
            if path.is_dir() and re.fullmatch(r"\d+", path.name)
        )

    return dirs


def main() -> int:
    """Run the converter from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert SkillCorner Open Data into a frame-level "
            "home/away training table."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Root directory containing match subdirectories.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output Parquet file.",
    )
    parser.add_argument(
        "--match-ids",
        nargs="+",
        default=None,
        help="Optional list of match directory IDs to convert.",
    )

    args = parser.parse_args()

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The report is deliberately written before Parquet generation and then
    # refreshed after every converted match. This guarantees that the CSV
    # exists even if a later conversion or Parquet write fails.
    report_path = output_path.with_name(
        "skillcorner_conversion_report_v3.csv"
    )
    reports: list[dict[str, Any]] = []

    def write_report() -> None:
        report_df = pd.DataFrame(reports, columns=REPORT_COLUMNS)
        report_df.to_csv(report_path, index=False, encoding="utf-8")

    print(f"[SKILLCORNER] Running script: {Path(__file__).resolve()}", flush=True)
    print(f"[SKILLCORNER] Working directory: {Path.cwd()}", flush=True)
    print(f"[SKILLCORNER] Report CSV target: {report_path}", flush=True)

    # Create a header-only report immediately, so its exact destination is
    # visible and the file remains available if conversion stops with error.
    write_report()
    print(
        f"[SKILLCORNER] Report CSV created: {report_path} "
        f"({len(REPORT_COLUMNS)} columns)",
        flush=True,
    )

    data_root = args.data_root.expanduser().resolve()
    match_dirs = discover_match_dirs(
        data_root,
        args.match_ids,
    )

    tables = []

    for match_dir in match_dirs:
        print(
            f"[SKILLCORNER] Converting match {match_dir.name} ...",
            flush=True,
        )

        try:
            table, report = convert_match(match_dir)
        except Exception:
            # Preserve the report for already completed matches and leave a
            # useful traceback in the console for the failing match.
            write_report()
            print(
                f"[SKILLCORNER] Conversion failed for match {match_dir.name}. "
                f"The report (possibly partial) is at: {report_path}",
                flush=True,
            )
            raise

        tables.append(table)
        reports.append(report)
        write_report()
        print(
            f"[SKILLCORNER] Report updated: {report_path} "
            f"({len(reports)} match row(s))",
            flush=True,
        )

    combined = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame()
    )

    # Write CSV before Parquet. A Parquet-engine or disk error must not
    # prevent the diagnostic report from being saved.
    write_report()
    combined.to_parquet(output_path, index=False)

    print("\n[COMPLETE]")
    print(
        f"Training Parquet: {output_path} "
        f"({len(combined):,} labelled frames)"
    )
    print(
        f"Conversion report: {report_path} "
        f"({len(REPORT_COLUMNS)} columns; overwritten)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())