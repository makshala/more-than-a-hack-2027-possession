#!/usr/bin/env python3
"""Convert SkillCorner Open Data tracking into a frame-level H/A training table.

The script intentionally does NOT modify the competition tracking/Parquet. It
creates a separate external-training dataset from SkillCorner's JSONL + match
metadata. It keeps SkillCorner's native 10 Hz sampling and computes a small set
of features that can also be computed from existing 111-column
Parquet under the same mathematical definitions.

Expected input layout (from https://github.com/SkillCorner/opendata):
    DATA_ROOT/
      1886347/
        1886347_match.json
        1886347_tracking_extrapolated.jsonl
      1899585/
        ...

Dependencies: numpy, pandas, scipy, pyarrow
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs

# ---------------------------------------------------------------------------
# Frozen physical constants set in tracking. Only the sampling interval
# changes for SkillCorner: 10 FPS => DT=0.1 s. Do not import competition config.
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

# Only features that have a clear mathematical counterpart in the current
# target feature pipeline are included in the first model contract.
MODEL_FEATURES_V1 = [
    "ball_speed",
    "home_near1_dist_m", "away_near1_dist_m",
    "home_near2_dist_m", "away_near2_dist_m",
    "home_near3_dist_m", "away_near3_dist_m",
    "home_in_3m", "away_in_3m",
    "home_in_5m", "away_in_5m",
    "home_near1_closing_speed_ms", "away_near1_closing_speed_ms",
    "home_near1_vel_cos", "away_near1_vel_cos",
    "home_min_tti_s", "away_min_tti_s", "tti_diff_s",
    "home_pitch_control", "is_contested",
]

DIAGNOSTIC_FEATURES = [
    "ball_is_detected",
    "home_n_detected", "away_n_detected",
    "home_n_tracked", "away_n_tracked",
    "home_centroid_vx", "home_centroid_vy",
    "away_centroid_vx", "away_centroid_vy",
]


def _id_key(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return np.nan
        out = float(value)
        return out if math.isfinite(out) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _as_timestamp_seconds(value: Any) -> float:
    """Parse numeric seconds or common HH:MM:SS[.tenths] timestamp strings."""
    numeric = _as_float(value)
    if np.isfinite(numeric):
        return numeric
    if isinstance(value, str) and ":" in value:
        try:
            parts = value.strip().split(":")
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
            if len(parts) == 2:
                minutes, seconds = parts
                return float(minutes) * 60.0 + float(seconds)
        except ValueError:
            pass
    return np.nan


def _normalise_possession_group(possession: Any) -> int:
    """Return 0 for home, 1 for away, -1 for unknown/unlabelled.

    The published SkillCorner tutorial uses the literal strings 'home team'
    and 'away team'. Short aliases are accepted defensively, but unknown values
    are never silently mapped to D or either team.
    """
    if not isinstance(possession, dict):
        return -1
    group = str(possession.get("group", "")).strip().lower().replace("_", " ")
    if group in {"home team", "home"}:
        return 0
    if group in {"away team", "away"}:
        return 1
    return -1


def _load_match_metadata(path: Path) -> tuple[dict[str, int], dict[str, Any], int]:
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    home_team_id = _id_key(meta.get("home_team", {}).get("id"))
    away_team_id = _id_key(meta.get("away_team", {}).get("id"))
    if not home_team_id or not away_team_id:
        raise ValueError(f"Missing home_team.id / away_team.id in {path}")

    player_team: dict[str, int] = {}
    for player in meta.get("players", []):
        pid = _id_key(player.get("id"))
        tid = _id_key(player.get("team_id"))
        if not pid:
            continue
        if tid == home_team_id:
            player_team[pid] = 1
        elif tid == away_team_id:
            player_team[pid] = 0

    periods = meta.get("match_periods", [])
    max_frame = max(
        (int(p.get("end_frame", 0) or 0) for p in periods),
        default=0,
    )
    if max_frame <= 0:
        raise ValueError(f"No usable match_periods/end_frame in {path}")
    if not any(v == 1 for v in player_team.values()) or not any(v == 0 for v in player_team.values()):
        raise ValueError(f"No roster players mapped to both teams in {path}")
    return player_team, meta, max_frame + 1


def _period_segments(period: np.ndarray) -> list[tuple[int, int, int]]:
    """Return contiguous [start, end) ranges for period 1 and 2."""
    segments: list[tuple[int, int, int]] = []
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if idx.size == 0:
            continue
        # Usually each half is one contiguous frame interval. Split defensively
        # if source frame ids contain a hole/other period inside the interval.
        starts = [int(idx[0])]
        ends: list[int] = []
        for j in range(1, len(idx)):
            if idx[j] != idx[j - 1] + 1:
                ends.append(int(idx[j - 1]) + 1)
                starts.append(int(idx[j]))
        ends.append(int(idx[-1]) + 1)
        segments.extend((p, s, e) for s, e in zip(starts, ends))
    return segments


def _player_velocity_like_team_pipeline(
    px: np.ndarray, py: np.ndarray, period: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the team's Savitzky-Golay/fallback velocity definition.

    Unlike the source ATD pipeline, SkillCorner provides XY but not the same
    provider speed field. We therefore compute kinematic velocities only. This
    matters when comparing near-player speed features; those are deliberately
    excluded from MODEL_FEATURES_V1 until exact equivalence is established.
    """
    n_frames, n_players = px.shape
    vx = np.zeros((n_frames, n_players), dtype=np.float32)
    vy = np.zeros((n_frames, n_players), dtype=np.float32)
    if n_players == 0:
        return vx, vy

    c_pos = savgol_coeffs(SAVGOL_WINDOW, SAVGOL_POLY, deriv=0, use="conv").astype(np.float32)
    c_vel = (savgol_coeffs(SAVGOL_WINDOW, SAVGOL_POLY, deriv=1, use="conv") / dt).astype(np.float32)
    kernel = np.ones(SAVGOL_WINDOW, dtype=np.float32)

    for _, start, end in _period_segments(period):
        x = px[start:end]
        y = py[start:end]
        valid = np.isfinite(x) & np.isfinite(y)
        x0 = np.where(valid, x, 0.0).astype(np.float32)
        y0 = np.where(valid, y, 0.0).astype(np.float32)
        has_full_window = convolve1d(
            valid.astype(np.float32), kernel, axis=0, mode="constant", cval=0.0
        ) > (SAVGOL_WINDOW - 0.5)

        xs = convolve1d(x0, c_pos, axis=0, mode="nearest")
        ys = convolve1d(y0, c_pos, axis=0, mode="nearest")
        x_smooth = np.where(has_full_window, xs, x).astype(np.float32)
        y_smooth = np.where(has_full_window, ys, y).astype(np.float32)

        vx_sg = convolve1d(x0, c_vel, axis=0, mode="nearest")
        vy_sg = convolve1d(y0, c_vel, axis=0, mode="nearest")

        dx_f = np.diff(x_smooth, axis=0, append=np.full((1, n_players), np.nan, dtype=np.float32))
        dy_f = np.diff(y_smooth, axis=0, append=np.full((1, n_players), np.nan, dtype=np.float32))
        dx_b = np.diff(x_smooth, axis=0, prepend=np.full((1, n_players), np.nan, dtype=np.float32))
        dy_b = np.diff(y_smooth, axis=0, prepend=np.full((1, n_players), np.nan, dtype=np.float32))
        has_f, has_b = np.isfinite(dx_f), np.isfinite(dx_b)

        vx_fb = np.where(
            has_f & has_b,
            0.5 * (dx_f + dx_b) / dt,
            np.where(has_f, dx_f / dt, np.where(has_b, dx_b / dt, 0.0)),
        )
        vy_fb = np.where(
            has_f & has_b,
            0.5 * (dy_f + dy_b) / dt,
            np.where(has_f, dy_f / dt, np.where(has_b, dy_b / dt, 0.0)),
        )
        vx_seg = np.where(has_full_window, vx_sg, vx_fb).astype(np.float32)
        vy_seg = np.where(has_full_window, vy_sg, vy_fb).astype(np.float32)
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
    """Approximate the team's 5-sample ball-kinematics definition.

    SkillCorner provides `is_detected_ball`; a finite but non-detected XY point
    is treated as an extrapolated/interpolated observation. Missing XY is
    linearly filled only internally to make derivatives computable, then the
    resulting velocity/speed is masked back to NaN on those rows.
    """
    n = len(bx)
    vx = np.full(n, np.nan, dtype=np.float32)
    vy = np.full(n, np.nan, dtype=np.float32)
    speed = np.full(n, np.nan, dtype=np.float32)
    c_vel = (savgol_coeffs(BALL_SAVGOL_WINDOW, BALL_SAVGOL_POLY, deriv=1, use="conv") / dt).astype(np.float32)
    kernel = np.ones(BALL_SAVGOL_WINDOW, dtype=np.float32)

    for _, start, end in _period_segments(period):
        x = bx[start:end].astype(np.float32, copy=True)
        y = by[start:end].astype(np.float32, copy=True)
        valid_xy = np.isfinite(x) & np.isfinite(y)
        if valid_xy.sum() < 2:
            continue
        idx = np.arange(len(x))
        x_fill = np.interp(idx, idx[valid_xy], x[valid_xy]).astype(np.float32)
        y_fill = np.interp(idx, idx[valid_xy], y[valid_xy]).astype(np.float32)

        # SkillCorner detected points -> status 0; finite extrapolated points
        # -> status 1; missing coordinates -> status 2. This mirrors the
        # target pipeline's observed/interpolated/latent distinction.
        status = np.where(valid_xy, np.where(ball_detected[start:end] == 1, 0, 1), 2).astype(np.int8)
        is_obs = (status <= 1).astype(np.float32)
        has_5 = (
            (convolve1d(is_obs, kernel, mode="constant", cval=0.0) > 4.5)
            | (convolve1d(is_obs, kernel, mode="constant", cval=0.0) < 0.5)
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
            np.where(has_f, dx_f / dt, np.where(has_b, dx_b / dt, 0.0)),
        )
        vy_fb = np.where(
            has_f & has_b,
            0.5 * (dy_f + dy_b) / dt,
            np.where(has_f, dy_f / dt, np.where(has_b, dy_b / dt, 0.0)),
        )
        vx_seg = np.where(has_5, vx_sg, vx_fb).astype(np.float32)
        vy_seg = np.where(has_5, vy_sg, vy_fb).astype(np.float32)
        speed_seg = np.hypot(vx_seg, vy_seg)
        over = speed_seg > BALL_MAX_SPEED_MS
        scale = BALL_MAX_SPEED_MS / np.maximum(speed_seg, 1e-5)
        vx_seg = np.where(over, vx_seg * scale, vx_seg).astype(np.float32)
        vy_seg = np.where(over, vy_seg * scale, vy_seg).astype(np.float32)
        speed_seg = np.minimum(speed_seg, BALL_MAX_SPEED_MS).astype(np.float32)

        vx_seg[~valid_xy] = np.nan
        vy_seg[~valid_xy] = np.nan
        speed_seg[~valid_xy] = np.nan
        vx[start:end], vy[start:end], speed[start:end] = vx_seg, vy_seg, speed_seg
    return vx, vy, speed


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
    n_frames = len(period)
    pvx, pvy = _player_velocity_like_team_pipeline(px, py, period, SKILLCORNER_DT)
    ball_vx, ball_vy, ball_speed = _ball_kinematics_like_team_pipeline(
        bx, by, ball_detected, period, SKILLCORNER_DT
    )

    out: dict[str, np.ndarray] = {
        "ball_speed": ball_speed,
        "ball_is_detected": ball_detected.astype(np.int8, copy=True),
        "home_n_detected": np.zeros(n_frames, dtype=np.int8),
        "away_n_detected": np.zeros(n_frames, dtype=np.int8),
        "home_n_tracked": np.zeros(n_frames, dtype=np.int8),
        "away_n_tracked": np.zeros(n_frames, dtype=np.int8),
    }
    # `detected` for players means detector-observed; position arrays retain
    # all valid tracked/extrapolated positions for distance and TTI features.
    ball_xy_valid = np.isfinite(bx) & np.isfinite(by)
    for code, team in ((1, "home"), (0, "away")):
        mask = player_team == code
        tx, ty = px[:, mask], py[:, mask]
        tvx, tvy = pvx[:, mask], pvy[:, mask]
        tdet = detected[:, mask]
        valid = np.isfinite(tx) & np.isfinite(ty)
        n_players = tx.shape[1]
        out[f"{team}_n_detected"] = np.sum(tdet & valid, axis=1).astype(np.int8)
        out[f"{team}_n_tracked"] = np.sum(valid, axis=1).astype(np.int8)

        # Team centroid motion (diagnostic only in V1; not a model feature yet).
        count = np.maximum(1, valid.sum(axis=1)).astype(np.float32)
        cx = np.where(valid, tx, 0.0).sum(axis=1) / count
        cy = np.where(valid, ty, 0.0).sum(axis=1) / count
        cx[valid.sum(axis=1) == 0] = np.nan
        cy[valid.sum(axis=1) == 0] = np.nan
        for p, s, e in _period_segments(period):
            if e - s >= 2:
                out[f"{team}_centroid_vx"] = out.get(f"{team}_centroid_vx", np.zeros(n_frames, dtype=np.float32))
                out[f"{team}_centroid_vy"] = out.get(f"{team}_centroid_vy", np.zeros(n_frames, dtype=np.float32))
                out[f"{team}_centroid_vx"][s:e] = np.gradient(cx[s:e], SKILLCORNER_DT)
                out[f"{team}_centroid_vy"][s:e] = np.gradient(cy[s:e], SKILLCORNER_DT)

        # If fewer than three roster slots exist, pad with an invalid sentinel.
        if n_players < 3:
            raise ValueError(f"Only {n_players} roster players found for {team}; expected at least 3")
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
            out[f"{team}_near{k+1}_dist_m"] = sorted_dist[:, k].astype(np.float32)

        # Counts within fixed physical radii, using all finite tracked positions.
        for radius, suffix in ((CONTROL_RADIUS_3M, "3m"), (SUPPORT_RADIUS_5M, "5m")):
            counts = np.sum(valid & (dist <= radius), axis=1).astype(np.float32)
            out[f"{team}_in_{suffix}"] = np.where(ball_xy_valid, counts, np.nan)

        near1_dist = sorted_dist[:, 0]
        n1_vx, n1_vy = sorted_vx[:, 0], sorted_vy[:, 0]
        n1_dx = np.take_along_axis(dx, order[:, :1], axis=1)[:, 0]
        n1_dy = np.take_along_axis(dy, order[:, :1], axis=1)[:, 0]
        n1_valid = np.isfinite(near1_dist) & np.isfinite(bx) & np.isfinite(by)
        # Same signed radial relative-velocity definition as tactics.py.
        rel_radial = ((n1_vx - 0.0) * n1_dx + (n1_vy - 0.0) * n1_dy) / (near1_dist + 0.1)
        # Subtract ball velocity below after both teams' nearest players are known.
        out[f"_{team}_near1_vx"] = np.where(n1_valid, n1_vx, np.nan)
        out[f"_{team}_near1_vy"] = np.where(n1_valid, n1_vy, np.nan)
        out[f"_{team}_near1_dx"] = np.where(n1_valid, n1_dx, np.nan)
        out[f"_{team}_near1_dy"] = np.where(n1_valid, n1_dy, np.nan)
        out[f"_{team}_near1_dist_tmp"] = np.where(n1_valid, near1_dist, np.nan)
        out[f"_{team}_near1_relradial_tmp"] = np.where(n1_valid, rel_radial, np.nan)
        out[f"_{team}_near1_player_speed_tmp"] = np.hypot(n1_vx, n1_vy)

        # Team-specific TTI: reaction delay + constant-speed travel from the
        # player's reaction-time projected location to the current ball point.
        proj_x = tx + tvx * REACTION_TIME_S
        proj_y = ty + tvy * REACTION_TIME_S
        tti = REACTION_TIME_S + np.hypot(bx[:, None] - proj_x, by[:, None] - proj_y) / PLAYER_MAX_SPEED_MS
        tti[~valid] = 99.0
        min_tti = np.minimum(np.min(tti, axis=1), 20.0).astype(np.float32)
        out[f"{team}_min_tti_s"] = np.where(ball_xy_valid, min_tti, np.nan)
        rates = np.where(valid & ball_xy_valid[:, None], np.exp(-tti / PITCH_CONTROL_TAU_S), 0.0).sum(axis=1)
        out[f"_{team}_pitch_control_rate"] = np.where(ball_xy_valid, rates, np.nan).astype(np.float32)

    for team in ("home", "away"):
        n1_vx = out[f"_{team}_near1_vx"]
        n1_vy = out[f"_{team}_near1_vy"]
        n1_dx = out[f"_{team}_near1_dx"]
        n1_dy = out[f"_{team}_near1_dy"]
        n1_dist = out[f"_{team}_near1_dist_tmp"]
        close = ((n1_vx - ball_vx) * n1_dx + (n1_vy - ball_vy) * n1_dy) / (n1_dist + 0.1)
        out[f"{team}_near1_closing_speed_ms"] = np.clip(close, -25.0, 25.0).astype(np.float32)
        player_speed = np.hypot(n1_vx, n1_vy)
        cos = (n1_vx * ball_vx + n1_vy * ball_vy) / (player_speed * ball_speed + 1e-5)
        cos_valid = (player_speed > 0.3) & (ball_speed > 0.3)
        cos_feature = np.where(cos_valid, np.clip(cos, -1.0, 1.0), 0.0)
        out[f"{team}_near1_vel_cos"] = np.where(
            np.isfinite(bx) & np.isfinite(by), cos_feature, np.nan
        ).astype(np.float32)

    out["near1_dist_diff_m"] = (out["away_near1_dist_m"] - out["home_near1_dist_m"]).astype(np.float32)
    out["tti_diff_s"] = (out["away_min_tti_s"] - out["home_min_tti_s"]).astype(np.float32)
    total_rate = out["_home_pitch_control_rate"] + out["_away_pitch_control_rate"]
    pc = np.where(total_rate > 1e-6, out["_home_pitch_control_rate"] / total_rate, 0.5)
    out["home_pitch_control"] = np.where(ball_xy_valid, pc, np.nan).astype(np.float32)
    contested = (
        (out["home_near1_dist_m"] <= CONTESTED_RADIUS_M)
        & (out["away_near1_dist_m"] <= CONTESTED_RADIUS_M)
    )
    out["is_contested"] = np.where(ball_xy_valid, contested.astype(np.float32), np.nan).astype(np.float32)

    # Cleanup internal helper arrays; they are not part of the training schema.
    for key in list(out):
        if key.startswith("_"):
            del out[key]
    return pd.DataFrame(out)


def convert_match(match_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    match_jsons = sorted(match_dir.glob("*_match.json"))
    tracking_files = sorted(match_dir.glob("*_tracking_extrapolated.jsonl"))
    if len(match_jsons) != 1 or len(tracking_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_match.json and one *_tracking_extrapolated.jsonl in {match_dir}; "
            f"found {len(match_jsons)} and {len(tracking_files)}"
        )
    match_json, tracking_path = match_jsons[0], tracking_files[0]
    match_id = match_json.name.split("_", 1)[0]
    player_team_map, metadata, n_frames = _load_match_metadata(match_json)

    all_player_ids = sorted(player_team_map.keys())
    player_slot = {pid: i for i, pid in enumerate(all_player_ids)}
    player_team = np.array([player_team_map[pid] for pid in all_player_ids], dtype=np.int8)
    n_players = len(all_player_ids)

    px = np.full((n_frames, n_players), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_players), np.nan, dtype=np.float32)
    detected = np.zeros((n_frames, n_players), dtype=bool)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    ball_detected = np.zeros(n_frames, dtype=np.int8)
    period = np.zeros(n_frames, dtype=np.int8)
    timestamp = np.full(n_frames, np.nan, dtype=np.float32)
    target = np.full(n_frames, -1, dtype=np.int8)

    unknown_player_ids: set[str] = set()
    n_json_frames = 0
    n_unknown_possession = 0
    n_ball_missing = 0
    n_frames_out_of_range = 0
    n_duplicate_frames = 0
    seen_frames: set[int] = set()

    with tracking_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"{tracking_path} is a Git LFS pointer, not the downloaded tracking data. "
                "Install Git LFS and run `git lfs pull` inside the SkillCorner repository."
            )
        f.seek(0)
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {tracking_path}:{line_no}: {exc}") from exc
            frame = int(rec.get("frame", -1))
            if frame < 0 or frame >= n_frames:
                n_frames_out_of_range += 1
                continue
            if frame in seen_frames:
                n_duplicate_frames += 1
            seen_frames.add(frame)
            n_json_frames += 1

            p = int(rec.get("period") or 0)
            period[frame] = p if p in (1, 2) else 0
            timestamp[frame] = _as_timestamp_seconds(rec.get("timestamp"))
            target[frame] = _normalise_possession_group(rec.get("possession"))
            if target[frame] < 0:
                n_unknown_possession += 1

            ball = rec.get("ball_data") or {}
            if isinstance(ball, dict):
                bx[frame] = _as_float(ball.get("x"))
                by[frame] = _as_float(ball.get("y"))
                ball_detected[frame] = int(bool(ball.get("is_detected", False)))
            if not (np.isfinite(bx[frame]) and np.isfinite(by[frame])):
                n_ball_missing += 1

            for player in rec.get("player_data") or []:
                if not isinstance(player, dict):
                    continue
                pid = _id_key(player.get("player_id"))
                slot = player_slot.get(pid)
                if slot is None:
                    if pid:
                        unknown_player_ids.add(pid)
                    continue
                x = _as_float(player.get("x"))
                y = _as_float(player.get("y"))
                if np.isfinite(x) and np.isfinite(y):
                    px[frame, slot] = x
                    py[frame, slot] = y
                    detected[frame, slot] = bool(player.get("is_detected", False))

    # SkillCorner coordinates are centred on the pitch. The distance-based
    # features below are translation invariant, so no coordinate shift is
    # needed. This avoids introducing a pitch-origin mismatch unnecessarily.
    feature_df = _compute_features(px, py, detected, player_team, bx, by, period, ball_detected)
    feature_df.insert(0, "timestamp_s", timestamp)
    feature_df.insert(0, "period", period)
    feature_df.insert(0, "frame", np.arange(n_frames, dtype=np.int32))
    feature_df.insert(0, "match_id", str(match_id))
    feature_df.insert(0, "target", target)

    # Keep only rows with a real H/A possession label and a valid match period.
    # Do NOT reinterpret unlabelled possession as D.
    labelled = feature_df[feature_df["target"].isin([0, 1]) & feature_df["period"].isin([1, 2])].copy()
    report = {
        "match_id": str(match_id),
        "match_name": f"{metadata.get('home_team', {}).get('name', '')} vs {metadata.get('away_team', {}).get('name', '')}",
        "tracking_file": tracking_path.name,
        "metadata_file": match_json.name,
        "json_frames_read": int(n_json_frames),
        "unique_frame_ids": int(len(seen_frames)),
        "duplicate_frame_ids": int(n_duplicate_frames),
        "out_of_range_frame_ids": int(n_frames_out_of_range),
        "labelled_home_frames": int((labelled["target"] == 0).sum()),
        "labelled_away_frames": int((labelled["target"] == 1).sum()),
        "unlabelled_or_unknown_possession_frames": int(n_unknown_possession),
        "labelled_frames_without_ball_speed": int((~np.isfinite(labelled["ball_speed"])).sum()),
        "frames_without_ball_xy_in_tracking": int(n_ball_missing),
        "unknown_player_ids_not_in_match_metadata": int(len(unknown_player_ids)),
        "unknown_player_id_examples": ",".join(sorted(unknown_player_ids)[:20]),
        "home_roster_players": int((player_team == 1).sum()),
        "away_roster_players": int((player_team == 0).sum()),
        "label_coverage_pct_of_json_frames": round(100.0 * len(labelled) / max(1, n_json_frames), 2),
        "ball_speed_valid_pct_of_labelled": round(100.0 * np.isfinite(labelled["ball_speed"]).mean(), 2) if len(labelled) else 0.0,
    }
    return labelled, report


def discover_match_dirs(data_root: Path, match_ids: Iterable[str] | None) -> list[Path]:
    if match_ids:
        dirs = [data_root / str(mid) for mid in match_ids]
    else:
        dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name))
    missing = [str(p) for p in dirs if not p.is_dir()]
    if missing:
        raise FileNotFoundError("Missing match directories: " + ", ".join(missing))
    if not dirs:
        raise FileNotFoundError(f"No numeric match directories found under {data_root}")
    return dirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Path to SkillCorner opendata/data/matches directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output Parquet path, e.g. artifacts/skillcorner_training.parquet")
    parser.add_argument("--match-ids", nargs="+", default=None,
                        help="Optional match IDs. Omit to process every numeric match directory.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional feature manifest JSON path")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional conversion report CSV path")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        parser.error(f"--data-root does not exist: {data_root}")
    match_dirs = discover_match_dirs(data_root, args.match_ids)

    tables: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for match_dir in match_dirs:
        print(f"[SKILLCORNER] Converting match {match_dir.name} ...", flush=True)
        table, report = convert_match(match_dir)
        tables.append(table)
        reports.append(report)
        print(
            f"  rows={len(table):,} | H={(table.target == 1).sum():,} | "
            f"A={(table.target == 0).sum():,} | ball_speed_valid={report['ball_speed_valid_pct_of_labelled']:.1f}%",
            flush=True,
        )

    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        combined.to_parquet(args.output, index=False)
    except ImportError as exc:
        raise RuntimeError("Parquet engine missing. Install pyarrow: pip install pyarrow") from exc

    report_path = args.report or args.output.with_name("skillcorner_conversion_report.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reports).to_csv(report_path, index=False)

    manifest_path = args.manifest or args.output.with_name("skillcorner_feature_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "SkillCorner Open Data external training table",
        "sampling_hz": SKILLCORNER_FPS,
        "dt_seconds": SKILLCORNER_DT,
        "pitch_length_m_reference": PITCH_LENGTH_M,
        "pitch_width_m_reference": PITCH_WIDTH_M,
        "coordinate_note": "SkillCorner XY is centred at pitch centre. Distance-based features are translation invariant.",
        "label_source": "tracking frame possession.group; 'home team' -> 1, 'away team' -> 0; all other values excluded",
        "target_column": "target",
        "target_mapping": {"home": 0, "away": 1},
        "physical_constants": {
            "reaction_time_s": REACTION_TIME_S,
            "player_max_speed_ms": PLAYER_MAX_SPEED_MS,
            "pitch_control_tau_s": PITCH_CONTROL_TAU_S,
            "contested_radius_m": CONTESTED_RADIUS_M,
            "control_radius_3m": CONTROL_RADIUS_3M,
            "support_radius_5m": SUPPORT_RADIUS_5M,
            "player_savgol_window": SAVGOL_WINDOW,
            "player_savgol_poly": SAVGOL_POLY,
        },
        "model_features_v1": MODEL_FEATURES_V1,
        "diagnostic_features": DIAGNOSTIC_FEATURES,
        "important_non_equivalences": [
            "SkillCorner player_data has is_detected but does not provide the same ATD player speed field; near-player provider-speed columns are intentionally not included in MODEL_FEATURES_V1.",
            "SkillCorner ball/player source quality differs from the competition tracking. Do not treat source-quality diagnostics as directly calibrated across datasets.",
            "No target-match fitting, scaling, calibration, or hyperparameter tuning is performed by this adapter.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n[COMPLETE]")
    print(f"Training Parquet: {args.output} ({len(combined):,} labelled frames)")
    print(f"Conversion report: {report_path}")
    print(f"Feature manifest: {manifest_path}")
    print("Reminder: split/train/validate by match_id, never randomly by frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
