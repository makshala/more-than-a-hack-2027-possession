from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

N_FRAMES: int = 199_319
PITCH_LENGTH: float = 105.0
PITCH_WIDTH: float = 68.0
STAGE1_PITCH_LENGTH: float = 105.3  # Per DATA.md and stage1_tracking.parquet specification
HALF_LENGTH: float = PITCH_LENGTH / 2.0  # 52.5 m
HALF_WIDTH: float = PITCH_WIDTH / 2.0    # 34.0 m


def _align_raw_atd_to_video_frames(atd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns 1-indexed ATD tracking frame_id (1..199319) to 0-indexed video frames (0..199318).
    """
    df = atd_df.copy()
    if "frame_id" in df.columns:
        min_fid = int(df["frame_id"].min())
        if min_fid == 1:
            df["video_frame"] = df["frame_id"].astype(np.int32) - 1
        else:
            df["video_frame"] = df["frame_id"].astype(np.int32)
        df = df.sort_values("video_frame").drop_duplicates("video_frame")
        full_index = pd.DataFrame({"video_frame": np.arange(N_FRAMES, dtype=np.int32)})
        df = full_index.merge(df, on="video_frame", how="left")
    return df.reset_index(drop=True)


def _extract_slot_arrays_from_raw_atd(
    atd_df: pd.DataFrame,
    home_slots: Optional[List[int]] = None,
    away_slots: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts (N, n_slots) position matrices scaled to [-52.5, 52.5] x [-34, 34]
    and speed matrices (m/s) from the 46-slot ATD DataFrame.
    """
    df = _align_raw_atd_to_video_frames(atd_df)
    cols = df.columns
    slot_x_cols, slot_y_cols, slot_s_cols = [], [], []
    h_indices, a_indices = [], []

    pattern = re.compile(r"^(?:(home|away)_)?(\d+)_x$", re.IGNORECASE)
    matched_slots = []
    for c in cols:
        m = pattern.match(str(c))
        if m:
            prefix, slot_num = m.group(1), int(m.group(2))
            base = f"{prefix}_{slot_num}" if prefix else f"{slot_num}"
            y_col = f"{base}_y"
            s_col = (
                f"{base}_s"
                if f"{base}_s" in cols
                else (f"{base}_speed" if f"{base}_speed" in cols else None)
            )
            if y_col in cols:
                matched_slots.append((slot_num, prefix, c, y_col, s_col))

    matched_slots.sort(key=lambda x: x[0])

    for idx, (slot_num, prefix, xc, yc, sc) in enumerate(matched_slots):
        slot_x_cols.append(xc)
        slot_y_cols.append(yc)
        slot_s_cols.append(sc)
        if prefix and prefix.lower() == "home":
            h_indices.append(idx)
        elif prefix and prefix.lower() == "away":
            a_indices.append(idx)
        elif home_slots is not None and slot_num in home_slots:
            h_indices.append(idx)
        elif away_slots is not None and slot_num in away_slots:
            a_indices.append(idx)
        else:
            if slot_num <= 23:
                h_indices.append(idx)
            else:
                a_indices.append(idx)

    x_norm = df[slot_x_cols].to_numpy(dtype=np.float32)
    y_norm = df[slot_y_cols].to_numpy(dtype=np.float32)

    max_x = float(np.nanmax(x_norm)) if x_norm.size > 0 else 1.0
    if max_x <= 1.5:
        x_m = (x_norm - 0.5) * PITCH_LENGTH
        y_m = (y_norm - 0.5) * PITCH_WIDTH
    else:
        x_m = x_norm * (PITCH_LENGTH / STAGE1_PITCH_LENGTH) - HALF_LENGTH
        y_m = y_norm - HALF_WIDTH

    if all(sc is not None for sc in slot_s_cols) and len(slot_s_cols) > 0:
        speeds = df[slot_s_cols].to_numpy(dtype=np.float32)
    else:
        vx = np.gradient(x_m, 0.04, axis=0)
        vy = np.gradient(y_m, 0.04, axis=0)
        speeds = np.hypot(vx, vy).astype(np.float32)

    bx_raw = df["ball_x"].to_numpy(dtype=np.float32)
    by_raw = df["ball_y"].to_numpy(dtype=np.float32)
    if float(np.nanmax(bx_raw)) <= 1.5:
        ball_x = (bx_raw - 0.5) * PITCH_LENGTH
        ball_y = (by_raw - 0.5) * PITCH_WIDTH
    else:
        ball_x = bx_raw * (PITCH_LENGTH / STAGE1_PITCH_LENGTH) - HALF_LENGTH
        ball_y = by_raw - HALF_WIDTH

    return (
        x_m[:, h_indices],
        y_m[:, h_indices],
        speeds[:, h_indices],
        x_m[:, a_indices],
        y_m[:, a_indices],
        speeds[:, a_indices],
        ball_x,
        ball_y,
    )


def _compute_mean_interteam_distance_chunked(
    hx: np.ndarray,
    hy: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    chunk_size: int = 16_384,
) -> np.ndarray:
    """Computes mean H-A inter-team distance in memory-safe chunks."""
    n_frames = hx.shape[0]
    mean_dist = np.full(n_frames, np.nan, dtype=np.float32)

    for start in range(0, n_frames, chunk_size):
        end = min(n_frames, start + chunk_size)
        dx = hx[start:end, :, None] - ax[start:end, None, :]
        dy = hy[start:end, :, None] - ay[start:end, None, :]
        dist_matrix = np.hypot(dx, dy)

        valid_pairs = ~np.isnan(dist_matrix)
        pair_counts = np.sum(valid_pairs, axis=(1, 2))
        dist_sum = np.nansum(dist_matrix, axis=(1, 2))

        valid_frames = pair_counts > 0
        chunk_res = np.full(end - start, np.nan, dtype=np.float32)
        chunk_res[valid_frames] = dist_sum[valid_frames] / pair_counts[valid_frames]
        mean_dist[start:end] = chunk_res

    nan_mask = np.isnan(mean_dist)
    if np.any(nan_mask):
        median_val = float(np.nanmedian(mean_dist)) if not np.all(nan_mask) else 22.0
        mean_dist[nan_mask] = median_val

    return mean_dist


def _masked_std_axis1(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Vectorized standard deviation along axis 1 without RuntimeWarnings."""
    count = np.sum(mask, axis=1)
    safe_count = np.maximum(count, 1)
    vals = np.where(mask, arr, 0.0)
    mean = np.sum(vals, axis=1) / safe_count
    sq_diff = np.where(mask, (arr - mean[:, None]) ** 2, 0.0)
    var = np.sum(sq_diff, axis=1) / safe_count
    std = np.sqrt(var)
    return np.where(count >= 3, std, 999.0).astype(np.float32)


def _detect_static_structures_from_slots(
    hx: np.ndarray,
    hy: np.ndarray,
    hs: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    as_: np.ndarray,
    ball_x: np.ndarray,
    ball_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detects static set-piece formations (free-kick defensive walls and corner box clusters)."""
    all_x = np.concatenate([hx, ax], axis=1)
    all_y = np.concatenate([hy, ay], axis=1)
    all_s = np.concatenate([hs, as_], axis=1)

    valid_s = ~np.isnan(all_s)
    s_count = np.maximum(np.sum(valid_s, axis=1), 1)
    mean_speed = np.nansum(all_s, axis=1) / s_count

    in_left_box = (all_x <= -36.0) & (np.abs(all_y) <= 20.16)
    in_right_box = (all_x >= 36.0) & (np.abs(all_y) <= 20.16)
    n_in_box = np.maximum(np.sum(in_left_box, axis=1), np.sum(in_right_box, axis=1))
    corner_box_cluster = (n_in_box >= 6) & (mean_speed < 2.2)

    dist_h_ball = np.hypot(hx - ball_x[:, None], hy - ball_y[:, None])
    dist_a_ball = np.hypot(ax - ball_x[:, None], ay - ball_y[:, None])

    h_in_wall_ring = (dist_h_ball >= 8.0) & (dist_h_ball <= 11.5) & (hs < 1.5)
    a_in_wall_ring = (dist_a_ball >= 8.0) & (dist_a_ball <= 11.5) & (as_ < 1.5)

    h_wall_compact = _masked_std_axis1(hy, h_in_wall_ring) < 2.8
    a_wall_compact = _masked_std_axis1(ay, a_in_wall_ring) < 2.8

    fk_wall = (
        ((np.sum(h_in_wall_ring, axis=1) >= 3) & h_wall_compact)
        | ((np.sum(a_in_wall_ring, axis=1) >= 3) & a_wall_compact)
    ) & (mean_speed < 2.0)

    return fk_wall, corner_box_cluster


def compute_kinematic_deadball_features(
    stage1_df: Optional[pd.DataFrame] = None,
    raw_atd_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Computes kinematic Dead-Ball features:
    - opponent_exclusion_score: FIFA distance rule around stationary/slow balls,
    - control_void_score: absence of registered ball control in Stage 1 state machine,
    - ball_death_flux: onset impulse for State D (ball crossing boundary or referee whistle).
    """
    if stage1_df is None and raw_atd_df is None:
        raise ValueError("At least one of stage1_df or raw_atd_df must be provided.")

    out = pd.DataFrame({"frame": np.arange(N_FRAMES, dtype=np.int32)})

    if raw_atd_df is not None:
        hx, hy, hs, ax, ay, as_, ball_x, ball_y = _extract_slot_arrays_from_raw_atd(raw_atd_df)
        all_speeds = np.concatenate([hs, as_], axis=1)
        valid_mask = ~np.isnan(all_speeds)
        n_visible = np.sum(valid_mask, axis=1)

        speed_sum = np.nansum(all_speeds, axis=1)
        mean_speed_all = np.where(n_visible > 0, speed_sum / np.maximum(n_visible, 1), 0.0).astype(
            np.float32
        )

        walking_count = np.sum((all_speeds < 2.0) & valid_mask, axis=1)
        walking_ratio_2ms = np.where(
            n_visible > 0, walking_count / np.maximum(n_visible, 1), 1.0
        ).astype(np.float32)

        no_pressing_dist_m = _compute_mean_interteam_distance_chunked(hx, hy, ax, ay)

        ball_out_of_bounds = (
            (np.abs(ball_x) > HALF_LENGTH) | (np.abs(ball_y) > HALF_WIDTH)
        ) & (~np.isnan(ball_x))

        fk_wall, corner_cluster = _detect_static_structures_from_slots(
            hx, hy, hs, ax, ay, as_, ball_x, ball_y
        )
        ball_x_centered = ball_x
        ball_y_centered = ball_y
    else:
        df = stage1_df
        mean_speed_all = df["all_players_mean_speed"].to_numpy(dtype=np.float32)

        walk_15 = df["players_walking_ratio"].to_numpy(dtype=np.float32)
        speed_factor = np.clip((2.0 - mean_speed_all) * 0.15, 0.0, 0.25)
        walking_ratio_2ms = np.clip(walk_15 + speed_factor, 0.0, 1.0).astype(np.float32)

        dx_c = df["home_centroid_x"].to_numpy(dtype=np.float32) - df["away_centroid_x"].to_numpy(
            dtype=np.float32
        )
        dy_c = df["home_centroid_y"].to_numpy(dtype=np.float32) - df["away_centroid_y"].to_numpy(
            dtype=np.float32
        )
        centroid_dist = np.hypot(dx_c, dy_c)
        var_x = (df["home_length_x"].to_numpy(dtype=np.float32) / 2.0) ** 2 + (
            df["away_length_x"].to_numpy(dtype=np.float32) / 2.0
        ) ** 2
        var_y = (df["home_width_y"].to_numpy(dtype=np.float32) / 2.0) ** 2 + (
            df["away_width_y"].to_numpy(dtype=np.float32) / 2.0
        ) ** 2
        no_pressing_dist_m = np.sqrt(centroid_dist**2 + var_x + var_y).astype(np.float32)

        # Fuse broadcast ball position when tactical ball is latent
        bx_m = df["ball_x"].to_numpy(dtype=np.float32)
        by_m = df["ball_y"].to_numpy(dtype=np.float32)
        if {"brd_ball_x", "brd_ball_y", "brd_valid", "ball_status_code"}.issubset(df.columns):
            brd_bx = df["brd_ball_x"].to_numpy(dtype=np.float32)
            brd_by = df["brd_ball_y"].to_numpy(dtype=np.float32)
            brd_ok = (
                (df["brd_valid"].to_numpy(dtype=np.int8) == 1)
                & (~np.isnan(brd_bx))
                & (~np.isnan(brd_by))
            )
            tac_latent = df["ball_status_code"].to_numpy(dtype=np.int8) > 0
            bx_m = np.where(tac_latent & brd_ok, brd_bx, bx_m)
            by_m = np.where(tac_latent & brd_ok, brd_by, by_m)

        ball_x_centered = (bx_m / STAGE1_PITCH_LENGTH - 0.5) * PITCH_LENGTH
        ball_y_centered = (by_m / PITCH_WIDTH - 0.5) * PITCH_WIDTH

        oob_geom = (np.abs(ball_x_centered) >= HALF_LENGTH - 0.15) | (
            np.abs(ball_y_centered) >= HALF_WIDTH - 0.15
        )
        near_bound = df["ball_near_boundary"].to_numpy(dtype=np.int8) > 0
        ball_out_of_bounds = oob_geom | (
            near_bound & (df["ball_speed"].to_numpy(dtype=np.float32) < 2.2)
        )

        if {"brd_valid", "brd_lookat_x", "brd_lookat_y", "brd_lookat_speed_ms"}.issubset(df.columns):
            b_val = df["brd_valid"].to_numpy(dtype=np.int8) == 1
            b_lx = (df["brd_lookat_x"].to_numpy(dtype=np.float32) / STAGE1_PITCH_LENGTH - 0.5) * PITCH_LENGTH
            b_ly = (df["brd_lookat_y"].to_numpy(dtype=np.float32) / PITCH_WIDTH - 0.5) * PITCH_WIDTH
            b_lspd = df["brd_lookat_speed_ms"].to_numpy(dtype=np.float32)
            tv_at_boundary = (
                b_val
                & (b_lspd < 2.2)
                & ((np.abs(b_lx) >= HALF_LENGTH - 3.5) | (np.abs(b_ly) >= HALF_WIDTH - 2.5))
                & (walking_ratio_2ms >= 0.55)
            )
            ball_out_of_bounds = ball_out_of_bounds | (tv_at_boundary & near_bound)

        in_box = df["players_in_penalty_box"].to_numpy(dtype=np.int8)
        ball_stat = df["ball_stationary_frames"].to_numpy(dtype=np.int16)

        corner_cluster = (in_box >= 6) & (ball_stat >= 10) & (walking_ratio_2ms >= 0.60)

        not_contested = df["is_contested"].to_numpy(dtype=np.int8) == 0
        h_n2 = df["home_near2_dist_m"].to_numpy(dtype=np.float32)
        a_n2 = df["away_near2_dist_m"].to_numpy(dtype=np.float32)
        wall_dist_proxy = ((h_n2 >= 7.5) & (h_n2 <= 12.0)) | ((a_n2 >= 7.5) & (a_n2 <= 12.0))
        fk_wall = (ball_stat >= 15) & not_contested & wall_dist_proxy & (walking_ratio_2ms >= 0.65)

    opponent_exclusion_score = np.zeros(N_FRAMES, dtype=np.float32)
    control_void_score = np.zeros(N_FRAMES, dtype=np.float32)
    ball_death_flux = np.zeros(N_FRAMES, dtype=np.float32)
    kickoff_setup_flag = np.zeros(N_FRAMES, dtype=bool)
    goal_kick_setup_flag = np.zeros(N_FRAMES, dtype=bool)

    if stage1_df is not None:
        df = stage1_df
        h_n1 = df["home_near1_dist_m"].to_numpy(dtype=np.float32)
        a_n1 = df["away_near1_dist_m"].to_numpy(dtype=np.float32)
        h_in3 = df["home_in_3m"].to_numpy(dtype=np.int8)
        a_in3 = df["away_in_3m"].to_numpy(dtype=np.int8)
        ball_stat_f = df["ball_stationary_frames"].to_numpy(dtype=np.float32)
        ball_spd = df["ball_speed"].to_numpy(dtype=np.float32)
        near_bound_f = df["ball_near_boundary"].to_numpy(dtype=np.int8) > 0
        is_cont = df["is_contested"].to_numpy(dtype=np.int8) > 0

        # 1. FIFA Opponent Exclusion Zone rule around stationary/slow ball
        min_d = np.minimum(h_n1, a_n1)
        max_d = np.maximum(h_n1, a_n1)
        one_team_in3 = ((h_in3 >= 1) & (a_in3 == 0)) | ((a_in3 >= 1) & (h_in3 == 0))
        opp_respects_dist = np.clip((max_d - 3.2) / 5.2, 0.0, 1.0)
        taker_ready = np.clip((3.2 - min_d) / 1.8, 0.0, 1.0)
        ball_slow_or_stat = np.clip(ball_stat_f / 12.0, 0.0, 1.0) * (ball_spd < 2.2)

        opponent_exclusion_score = (
            opp_respects_dist * (0.6 * taker_ready + 0.4 * one_team_in3) * ball_slow_or_stat
        ).astype(np.float32)

        # 2. Control Void Score from Stage 1 possession state machine
        if "frames_since_last_control" in df.columns:
            fslc = df["frames_since_last_control"].to_numpy(dtype=np.float32)
            control_void_score = (
                np.clip((fslc - 18.0) / 60.0, 0.0, 1.0)
                * np.clip((walking_ratio_2ms - 0.28) / 0.45, 0.0, 1.0)
            ).astype(np.float32)

        # 3. Specific static structures: Kick-off and Goal Kick
        if {"home_centroid_x_norm", "away_centroid_x_norm"}.issubset(df.columns):
            h_cxn = df["home_centroid_x_norm"].to_numpy(dtype=np.float32)
            a_cxn = df["away_centroid_x_norm"].to_numpy(dtype=np.float32)
            ball_near_center = (np.abs(ball_x_centered) < 3.5) & (np.abs(ball_y_centered) < 3.5)
            halves_separated = (h_cxn < 51.5) & (a_cxn > 53.5)
            kickoff_setup_flag = ball_near_center & halves_separated & (walking_ratio_2ms > 0.65)

        if {"home_gk_dist_m", "away_gk_dist_m"}.issubset(df.columns):
            gk_near = np.minimum(
                df["home_gk_dist_m"].to_numpy(dtype=np.float32),
                df["away_gk_dist_m"].to_numpy(dtype=np.float32),
            ) < 4.0
            in_goal_area = (np.abs(ball_x_centered) >= HALF_LENGTH - 7.5) & (
                np.abs(ball_y_centered) <= 11.0
            )
            goal_kick_setup_flag = gk_near & in_goal_area & (ball_stat_f >= 10) & (max_d > 8.0)

        # 4. Ball Death Flux (onset of State D - boundary crossing or referee whistle after foul)
        oob_int = (ball_out_of_bounds | near_bound_f).astype(np.int8)
        oob_onset = (oob_int == 1) & (np.roll(oob_int, 1) == 0)
        oob_onset[0] = False
        oob_flux = np.convolve(oob_onset.astype(np.float32), np.hanning(11), mode="same")

        recent_contest = (
            np.convolve(is_cont.astype(np.float32), np.ones(35, dtype=np.float32) / 35.0, mode="same")
            > 0.12
        )
        speed_drop = np.maximum(0.0, np.roll(mean_speed_all, 25) - mean_speed_all)
        speed_drop[:25] = 0.0
        whistle_flux = np.where(
            recent_contest,
            np.clip(speed_drop / 1.4, 0.0, 1.0) * np.clip(walking_ratio_2ms, 0.0, 1.0),
            0.0,
        )

        ball_death_flux = np.clip(0.80 * oob_flux + 0.60 * whistle_flux, 0.0, 1.5).astype(np.float32)

    static_structure_flag = (
        fk_wall | corner_cluster | kickoff_setup_flag | goal_kick_setup_flag
    ).astype(bool)

    out["mean_speed_all"] = mean_speed_all
    out["walking_ratio_2ms"] = walking_ratio_2ms
    out["no_pressing_dist_m"] = no_pressing_dist_m
    out["ball_out_of_bounds"] = ball_out_of_bounds.astype(bool)
    out["ball_x_centered"] = ball_x_centered.astype(np.float32)
    out["ball_y_centered"] = ball_y_centered.astype(np.float32)
    out["free_kick_wall_flag"] = fk_wall.astype(bool)
    out["corner_box_cluster_flag"] = corner_cluster.astype(bool)
    out["static_structure_flag"] = static_structure_flag
    out["opponent_exclusion_score"] = opponent_exclusion_score
    out["control_void_score"] = control_void_score
    out["ball_death_flux"] = ball_death_flux

    # Composite kinematic Dead-Ball score
    s_speed = np.clip((3.3 - mean_speed_all) / 2.0, 0.0, 1.0)
    s_walk = np.clip((walking_ratio_2ms - 0.32) / 0.50, 0.0, 1.0)
    s_press = np.clip((no_pressing_dist_m - 14.0) / 14.0, 0.0, 1.0)
    s_oob = ball_out_of_bounds.astype(np.float32) * 0.46
    s_struct = static_structure_flag.astype(np.float32) * 0.30
    s_excl = opponent_exclusion_score * 0.32
    s_void = control_void_score * 0.28

    raw_score = (
        0.25 * s_speed
        + 0.30 * s_walk
        + 0.12 * s_press
        + s_oob
        + s_struct
        + s_excl
        + s_void
    )

    if stage1_df is not None:
        stat_norm = np.clip(
            stage1_df["ball_stationary_frames"].to_numpy(dtype=np.float32) / 30.0, 0.0, 1.0
        )
        uncontested = 1.0 - stage1_df["is_contested"].to_numpy(dtype=np.float32)

        if "ball_to_brd_lookat_dist_m" in stage1_df.columns and "brd_valid" in stage1_df.columns:
            look_dist = stage1_df["ball_to_brd_lookat_dist_m"].to_numpy(dtype=np.float32)
            b_val = stage1_df["brd_valid"].to_numpy(dtype=np.int8) == 1
            tac_inferred = stage1_df["ball_status_code"].to_numpy(dtype=np.int8) == 2
            stat_trust = np.where(b_val & tac_inferred & (look_dist > 18.0), 0.35, 1.0).astype(
                np.float32
            )
            stat_norm *= stat_trust

        raw_score = 0.65 * raw_score + 0.25 * stat_norm + 0.10 * uncontested

        # Suppress only during immediate active multi-touch chains (fslc <= 10 frames = 0.4s)
        # so foul/out-of-bounds onset right after a touch is not suppressed.
        if {"home_consecutive_touches", "away_consecutive_touches", "frames_since_last_control"}.issubset(
            stage1_df.columns
        ):
            max_touches = np.maximum(
                stage1_df["home_consecutive_touches"].to_numpy(dtype=np.int16),
                stage1_df["away_consecutive_touches"].to_numpy(dtype=np.int16),
            )
            fslc = stage1_df["frames_since_last_control"].to_numpy(dtype=np.int32)
            active_possession_chain = (max_touches >= 2) & (fslc <= 10) & (~ball_out_of_bounds)
            raw_score = np.where(active_possession_chain, raw_score * 0.45, raw_score)

        if "brd_is_replay" in stage1_df.columns:
            brd_rep = stage1_df["brd_is_replay"].to_numpy(dtype=np.int8) == 1
            players_slow = mean_speed_all < 2.65
            raw_score = np.where(brd_rep & players_slow, np.maximum(raw_score, 0.88), raw_score)

    kernel = np.ones(15, dtype=np.float32) / 15.0
    smooth_score = np.convolve(np.clip(raw_score, 0.0, 1.0), kernel, mode="same")
    out["kinematic_d_score"] = np.clip(smooth_score, 0.0, 1.0).astype(np.float32)

    return out