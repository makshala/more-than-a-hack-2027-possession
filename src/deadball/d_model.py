from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.deadball.kinematic_d import (
    HALF_LENGTH,
    HALF_WIDTH,
    compute_kinematic_deadball_features,
)
from src.deadball.smart_tags import N_FRAMES, build_smart_tag_priors


def _compute_restart_impulse_signal(
    stage1_df: pd.DataFrame,
    restart_hint: np.ndarray,
    live_tag_onset_signal: np.ndarray,
) -> np.ndarray:
    """
    Computes the frame-level restart impulse strength (phi_{D -> L}, "frame 0").
    Integrates 6 physical and tactical signals:
    1. Ball acceleration and speed jump (tactical or broadcast fallback) after a stationary phase.
    2. Speed burst of the set-piece taker near the ball.
    3. Reset of frames_since_last_control (transition from >= 22 frames without control down to <= 1).
    4. Broadcast camera whip (brd_lookat_speed_ms) and tactical camera pan acceleration.
    5. Lead/Lag boundary features (near1_diff_leadlag_10f, pitch_ctrl_leadlag_10f).
    6. Onset impulse of a live tactical phase tag (BUILD UP, LONG BALL, PROGRESSION).
    """
    ball_speed = stage1_df["ball_speed"].to_numpy(dtype=np.float32)
    ball_acc = stage1_df["ball_acc"].to_numpy(dtype=np.float32)
    ball_stat = stage1_df["ball_stationary_frames"].to_numpy(dtype=np.float32)
    deflection = stage1_df["ball_deflection_angle"].to_numpy(dtype=np.float32)
    status_code = stage1_df["ball_status_code"].to_numpy(dtype=np.int8)

    ball_reliability = np.where(
        status_code == 0, 1.0, np.where(status_code == 1, 0.70, 0.15)
    ).astype(np.float32)

    # Fuse broadcast ball coordinates when tactical ball tracking is in a gap (status_code > 0)
    if {"brd_ball_x", "brd_ball_y", "brd_valid"}.issubset(stage1_df.columns):
        brd_bx = stage1_df["brd_ball_x"].to_numpy(dtype=np.float32)
        brd_by = stage1_df["brd_ball_y"].to_numpy(dtype=np.float32)
        brd_ok = (
            (stage1_df["brd_valid"].to_numpy(dtype=np.int8) == 1)
            & (~np.isnan(brd_bx))
            & (~np.isnan(brd_by))
        )
        brd_vx = np.gradient(np.nan_to_num(brd_bx, nan=0.0), 0.04)
        brd_vy = np.gradient(np.nan_to_num(brd_by, nan=0.0), 0.04)
        brd_spd = np.clip(np.hypot(brd_vx, brd_vy), 0.0, 38.0).astype(np.float32)
        brd_acc = np.clip(np.abs(np.gradient(brd_spd, 0.04)), 0.0, 150.0).astype(np.float32)

        use_brd_ball = (status_code > 0) & brd_ok
        ball_speed = np.where(use_brd_ball, brd_spd, ball_speed)
        ball_acc = np.where(use_brd_ball, brd_acc, ball_acc)
        ball_reliability = np.where(use_brd_ball, 0.85, ball_reliability).astype(np.float32)

    prev_stat = np.roll(ball_stat, 1)
    prev_stat[0] = 0.0
    was_stationary = np.clip(prev_stat / 8.0, 0.20, 1.0)

    speed_jump = np.maximum(0.0, ball_speed - np.roll(ball_speed, 2))
    speed_jump[:2] = 0.0

    ball_kick_impulse = (
        0.45 * np.clip(ball_acc / 35.0, 0.0, 1.5)
        + 0.35 * np.clip(speed_jump / 8.0, 0.0, 1.5)
        + 0.20 * np.clip(deflection / 1.5, 0.0, 1.0)
    ) * was_stationary * ball_reliability

    # Set-piece taker speed burst near the ball
    h_dist = stage1_df["home_near1_dist_m"].to_numpy(dtype=np.float32)
    a_dist = stage1_df["away_near1_dist_m"].to_numpy(dtype=np.float32)
    h_spd = stage1_df["home_near1_speed_ms"].to_numpy(dtype=np.float32)
    a_spd = stage1_df["away_near1_speed_ms"].to_numpy(dtype=np.float32)
    h_close = stage1_df["home_near1_closing_speed_ms"].to_numpy(dtype=np.float32)
    a_close = stage1_df["away_near1_closing_speed_ms"].to_numpy(dtype=np.float32)

    use_home = (restart_hint == "H") | ((restart_hint != "A") & (h_dist <= a_dist))
    taker_dist = np.where(use_home, h_dist, a_dist)
    taker_spd = np.where(use_home, h_spd, a_spd)
    taker_close = np.where(use_home, h_close, a_close)

    taker_acc = np.maximum(0.0, taker_spd - np.roll(taker_spd, 3))
    taker_acc[:3] = 0.0

    near_ball_factor = np.clip((3.2 - taker_dist) / 2.2, 0.0, 1.0)
    player_burst_impulse = (
        0.55 * np.clip(taker_acc / 2.5, 0.0, 1.2)
        + 0.45 * np.clip(taker_close / 4.0, 0.0, 1.2)
    ) * near_ball_factor

    total_impulse = ball_kick_impulse + (1.0 - 0.55 * ball_reliability) * player_burst_impulse

    # Direct ball-contact reset after a stoppage gap
    if "frames_since_last_control" in stage1_df.columns:
        fslc = stage1_df["frames_since_last_control"].to_numpy(dtype=np.int32)
        prev_fslc = np.roll(fslc, 1)
        prev_fslc[0] = 0
        contact_reset = (fslc <= 1) & (prev_fslc >= 22)
        reset_strength = np.where(
            contact_reset, np.clip(prev_fslc.astype(np.float32) / 90.0, 0.4, 1.2), 0.0
        )
        reset_smooth = np.convolve(
            reset_strength, np.array([0.25, 0.6, 1.0, 0.6, 0.25], dtype=np.float32), mode="same"
        )
        total_impulse += 0.55 * reset_smooth

    # Broadcast camera whip on long restarts
    if {"brd_lookat_speed_ms", "brd_valid"}.issubset(stage1_df.columns):
        brd_lspd = stage1_df["brd_lookat_speed_ms"].to_numpy(dtype=np.float32)
        brd_val = stage1_df["brd_valid"].to_numpy(dtype=np.int8) == 1
        if "frames_since_brd_replay" in stage1_df.columns:
            fs_rep = stage1_df["frames_since_brd_replay"].to_numpy(dtype=np.int32)
            brd_val = brd_val & (fs_rep > 5)

        cam_jerk = np.maximum(0.0, brd_lspd - np.roll(brd_lspd, 3))
        cam_jerk[:3] = 0.0
        cam_whip_signal = np.where(
            brd_val,
            0.22 * np.clip(brd_lspd / 12.0, 0.0, 1.2) + 0.18 * np.clip(cam_jerk / 6.0, 0.0, 1.2),
            0.0,
        )
        total_impulse += cam_whip_signal

    # Tactical camera pan acceleration
    if "camera_pan_speed" in stage1_df.columns:
        tac_pan = stage1_df["camera_pan_speed"].to_numpy(dtype=np.float32)
        tac_pan_jump = np.maximum(0.0, tac_pan - np.roll(tac_pan, 4))
        tac_pan_jump[:4] = 0.0
        total_impulse += 0.12 * np.clip(tac_pan_jump / 180.0, 0.0, 1.0)

    # Lead/Lag edge features (10 frames)
    if {"near1_diff_leadlag_10f", "pitch_ctrl_leadlag_10f"}.issubset(stage1_df.columns):
        n1_ll10 = np.abs(stage1_df["near1_diff_leadlag_10f"].to_numpy(dtype=np.float32))
        pc_ll10 = np.abs(stage1_df["pitch_ctrl_leadlag_10f"].to_numpy(dtype=np.float32))
        edge_signal = 0.20 * np.clip(n1_ll10 / 4.0, 0.0, 1.0) + 0.20 * np.clip(
            pc_ll10 / 0.35, 0.0, 1.0
        )
        total_impulse += edge_signal

    # Live phase tag onset impulse
    total_impulse += 0.25 * live_tag_onset_signal

    return total_impulse.astype(np.float32)


def _compute_type_geometry_compatibility(
    sp_type_val: Optional[str],
    bx_slice: np.ndarray,
    by_slice: np.ndarray,
) -> np.ndarray:
    """
    Returns a spatial compatibility multiplier [0.5 .. 1.35] for a candidate restart frame
    based on the set-piece type (throw_in, corner_kick, goal_kick, kick_off).
    """
    n = len(bx_slice)
    if sp_type_val is None or n == 0:
        return np.ones(n, dtype=np.float32)

    if sp_type_val == "throw_in":
        near_sideline = np.abs(by_slice) >= (HALF_WIDTH - 4.5)
        return np.where(near_sideline, 1.30, 0.65).astype(np.float32)

    if sp_type_val == "corner_kick":
        near_corner = (np.abs(bx_slice) >= (HALF_LENGTH - 6.5)) & (
            np.abs(by_slice) >= (HALF_WIDTH - 6.5)
        )
        return np.where(near_corner, 1.35, 0.60).astype(np.float32)

    if sp_type_val == "goal_kick":
        in_goal_box = (np.abs(bx_slice) >= (HALF_LENGTH - 9.5)) & (np.abs(by_slice) <= 12.5)
        return np.where(in_goal_box, 1.35, 0.65).astype(np.float32)

    if sp_type_val in ("kick_off", "post_goal"):
        near_center = (np.abs(bx_slice) <= 5.0) & (np.abs(by_slice) <= 5.0)
        return np.where(near_center, 1.35, 0.60).astype(np.float32)

    return np.ones(n, dtype=np.float32)


def _snap_both_deadball_edges(
    p_dead_raw: np.ndarray,
    is_sp_window: np.ndarray,
    sp_type_arr: np.ndarray,
    death_flux: np.ndarray,
    restart_flux: np.ndarray,
    kinematic_d_score: np.ndarray,
    ball_x_centered: np.ndarray,
    ball_y_centered: np.ndarray,
) -> np.ndarray:
    """
    Snaps both boundaries of each Dead-Ball window without prematurely truncating D segments.
    """
    p_refined = p_dead_raw.copy()

    padded = np.concatenate([[False], is_sp_window, [False]])
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1]) - 1

    for s_idx, e_idx in zip(starts, ends):
        win_len = e_idx - s_idx + 1
        if win_len < 12:
            continue

        win_types = sp_type_arr[s_idx : e_idx + 1]
        valid_types = [t for t in win_types if t is not None]
        dom_type = valid_types[0] if len(valid_types) > 0 else None

        # --- STEP A: Detect right boundary ("frame 0" - actual ball restart) ---
        search_start = s_idx + int(0.45 * win_len)
        search_slice = slice(search_start, e_idx + 1)

        local_imp = restart_flux[search_slice]
        pre_kin = kinematic_d_score[search_slice]
        post_kin = np.roll(kinematic_d_score, -25)[search_slice]
        kin_drop = np.maximum(0.0, pre_kin - post_kin)

        geom_mult = _compute_type_geometry_compatibility(
            dom_type,
            ball_x_centered[search_slice],
            ball_y_centered[search_slice],
        )

        restart_score = (local_imp * (0.5 + 0.5 * pre_kin) + 0.45 * kin_drop) * geom_mult

        if restart_score.size > 0 and np.max(restart_score) > 0.34:
            best_offset = int(np.argmax(restart_score))
            frame_zero = search_start + best_offset
        else:
            frame_zero = max(search_start, e_idx - 50)

        # --- STEP B: Detect left boundary (d_start - ball out-of-bounds or whistle) ---
        left_search_end = max(s_idx + 1, frame_zero - 12)
        left_slice = slice(s_idx, left_search_end)
        local_death = death_flux[left_slice]
        local_kin = kinematic_d_score[left_slice]
        future_kin = np.roll(kinematic_d_score, -25)[left_slice]

        death_candidates = np.flatnonzero(
            ((local_death > 0.22) & (future_kin > 0.30))
            | (local_kin > 0.34)
            | (p_dead_raw[left_slice] > 0.46)
        )
        if death_candidates.size > 0:
            d_start = s_idx + int(death_candidates[0])
            refine_s = max(s_idx, d_start - 12)
            refine_e = min(frame_zero - 5, d_start + 12)
            if refine_e > refine_s and np.max(death_flux[refine_s:refine_e]) > 0.25:
                d_start = refine_s + int(np.argmax(death_flux[refine_s:refine_e]))
        else:
            d_start = s_idx

        # ZMIANA: Delikatniejsze tłumienie przed gwizdkiem (z 0.42 na 0.85) - nie ucinamy akcji!
        if d_start > s_idx:
            p_refined[s_idx:d_start] = np.minimum(
                p_refined[s_idx:d_start],
                0.85 * kinematic_d_score[s_idx:d_start],
            )

        if frame_zero > d_start:
            p_refined[d_start:frame_zero] = np.maximum(
                p_refined[d_start:frame_zero],
                np.clip(0.64 + 0.34 * kinematic_d_score[d_start:frame_zero], 0.68, 0.99),
            )

        # ZMIANA: Drastyczne skrócenie "ogona" martwej piłki po wznowieniu z 22 klatek na 3 klatki!
        post_end = min(N_FRAMES, frame_zero + 3)
        decay_len = post_end - frame_zero
        if decay_len > 0:
            decay_curve = np.linspace(0.05, 0.15, decay_len, dtype=np.float32)
            p_refined[frame_zero:post_end] = np.minimum(
                p_refined[frame_zero:post_end],
                decay_curve * kinematic_d_score[frame_zero:post_end],
            )

    return p_refined


def _viterbi_and_posterior_decode(
    p_dead_snapped: np.ndarray,
    death_flux: np.ndarray,
    restart_flux: np.ndarray,
    period_arr: np.ndarray,
    base_switch_penalty: float = 8.5,
    min_live_gap_frames: int = 22,
    min_dead_duration_frames: int = 25,
) -> np.ndarray:
    """
    Global 2-state Log-Odds Viterbi decoder coupled with posterior probability calibration.
    Uses responsive transition penalties (base_switch_penalty=8.5, min_dead_duration_frames=25 = 1.0s)
    so genuine quick throw-ins and free kicks are preserved to separate consecutive possessions.
    """
    p_clamped = np.clip(p_dead_snapped, 1e-4, 1.0 - 1e-4).astype(np.float32)
    log_odds_d = np.log(p_clamped / (1.0 - p_clamped)).astype(np.float32)

    cost_l2d = np.maximum(1.2, base_switch_penalty - 6.0 * np.clip(death_flux, 0.0, 1.4)).astype(
        np.float32
    )
    cost_d2l = np.maximum(1.2, base_switch_penalty - 6.5 * np.clip(restart_flux, 0.0, 1.4)).astype(
        np.float32
    )

    viterbi_state = np.zeros(N_FRAMES, dtype=bool)

    for p_val in (1, 2):
        idx = np.flatnonzero(period_arr == p_val)
        if idx.size == 0:
            continue
        s_f, e_f = int(idx[0]), int(idx[-1])
        length = e_f - s_f + 1

        emit_list = log_odds_d[s_f : e_f + 1].tolist()
        c_l2d_list = cost_l2d[s_f : e_f + 1].tolist()
        c_d2l_list = cost_d2l[s_f : e_f + 1].tolist()

        back_0 = [0] * length
        back_1 = [0] * length

        v0 = -0.5 * emit_list[0]
        v1 = 0.5 * emit_list[0]

        for t in range(1, length):
            em = 0.5 * emit_list[t]
            # State 0 (LIVE)
            stay_0 = v0 - em
            from_1 = v1 - c_d2l_list[t] - em
            if stay_0 >= from_1:
                nv0 = stay_0
                back_0[t] = 0
            else:
                nv0 = from_1
                back_0[t] = 1

            # State 1 (DEAD)
            stay_1 = v1 + em
            from_0 = v0 - c_l2d_list[t] + em
            if stay_1 >= from_0:
                nv1 = stay_1
                back_1[t] = 1
            else:
                nv1 = from_0
                back_1[t] = 0

            v0, v1 = nv0, nv1

        path = [0] * length
        curr = 1 if v1 > v0 else 0
        path[length - 1] = curr
        for t in range(length - 1, 0, -1):
            curr = back_1[t] if curr == 1 else back_0[t]
            path[t - 1] = curr

        viterbi_state[s_f : e_f + 1] = np.array(path, dtype=bool)

    # Morphological cleanup preserving all stoppages >= 1.0s (25 frames)
    padded_inv = np.concatenate([[False], ~viterbi_state, [False]])
    g_starts = np.flatnonzero(padded_inv[1:] & ~padded_inv[:-1])
    g_ends = np.flatnonzero(~padded_inv[1:] & padded_inv[:-1]) - 1
    for gs, ge in zip(g_starts, g_ends):
        if gs > 0 and ge < N_FRAMES - 1 and (ge - gs + 1) < min_live_gap_frames:
            viterbi_state[gs : ge + 1] = True

    padded_d = np.concatenate([[False], viterbi_state, [False]])
    d_starts = np.flatnonzero(padded_d[1:] & ~padded_d[:-1])
    d_ends = np.flatnonzero(~padded_d[1:] & padded_d[:-1]) - 1
    for ds, de in zip(d_starts, d_ends):
        if (de - ds + 1) < min_dead_duration_frames:
            viterbi_state[ds : de + 1] = False

    # Posterior probability calibration aligned with Viterbi path:
    # Inside Viterbi-DEAD: [0.60, 0.99]
    # Inside Viterbi-LIVE: preserve soft boundary evidence up to 0.48
    p_calibrated = np.where(
        viterbi_state,
        np.clip(0.60 + 0.39 * p_dead_snapped, 0.60, 0.99),
        np.clip(0.65 * p_dead_snapped, 0.01, 0.48),
    ).astype(np.float32)

    return p_calibrated


def evaluate_deadball_on_public_gt(
    result_df: pd.DataFrame,
    public_gt_path: Union[str, Path] = "evaluation/gt/BAR-ATH_possession_gt_public.csv",
) -> Optional[Dict[str, float]]:
    """
    Evaluates Dead-Ball detection quality on the 5-minute public sanity slice
    (frames 11185..18684, strictly read-only pipeline verification per EVALUATION.md).
    """
    gt_path = Path(public_gt_path)
    if not gt_path.exists():
        return None

    gt_df = pd.read_csv(gt_path)
    label_col = [c for c in gt_df.columns if c.strip().lower() == "label"][0]
    gt_labels = gt_df[label_col].astype(str).str.strip().str.upper().to_numpy()

    graded_mask = np.isin(gt_labels, ["H", "A", "D"])
    if not np.any(graded_mask):
        return None

    true_d = gt_labels[graded_mask] == "D"
    pred_d = result_df["p_dead"].to_numpy()[graded_mask] >= 0.50

    tp = int(np.sum(true_d & pred_d))
    fp = int(np.sum((~true_d) & pred_d))
    fn = int(np.sum(true_d & (~pred_d)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_d = 2.0 * precision * recall / max(precision + recall, 1e-8)

    pad_true = np.concatenate([[False], true_d, [False]])
    n_true_spells = int(np.sum(pad_true[1:] & ~pad_true[:-1]))

    pad_pred = np.concatenate([[False], pred_d, [False]])
    n_pred_spells = int(np.sum(pad_pred[1:] & ~pad_pred[:-1]))

    metrics = {
        "d_precision": round(precision, 4),
        "d_recall": round(recall, 4),
        "d_f1": round(f1_d, 4),
        "true_d_frames": int(np.sum(true_d)),
        "pred_d_frames": int(np.sum(pred_d)),
        "true_d_segments": n_true_spells,
        "pred_d_segments": n_pred_spells,
    }
    return metrics


def run_deadball_pipeline(
    stage1_parquet_path: Union[str, Path] = "data/interim/stage1_tracking.parquet",
    smart_tags_xml_path: Optional[Union[str, Path]] = None,
    raw_atd_df: Optional[pd.DataFrame] = None,
    output_parquet_path: Union[str, Path] = "data/interim/stage2_deadball.parquet",
    public_gt_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Main Role 3 Dead-Ball estimation pipeline.
    Produces calibrated out-of-play probabilities P(D) and set-piece context columns.
    """
    stage1_path = Path(stage1_parquet_path)
    if not stage1_path.exists():
        raise FileNotFoundError(f"Stage 1 input file not found: {stage1_path}")

    stage1_df = pd.read_parquet(stage1_path)
    if len(stage1_df) != N_FRAMES:
        raise ValueError(
            f"Expected {N_FRAMES} frames in stage1_tracking.parquet, got {len(stage1_df)}."
        )

    if "frame_id" in stage1_df.columns:
        stage1_df = stage1_df.sort_values("frame_id").reset_index(drop=True)

    priors_df = build_smart_tag_priors(
        xml_path=smart_tags_xml_path,
        stage1_df=stage1_df,
    )

    kin_df = compute_kinematic_deadball_features(
        stage1_df=stage1_df,
        raw_atd_df=raw_atd_df,
    )

    is_sp_window = priors_df["is_set_piece_window"].to_numpy(dtype=bool)
    sp_prior_prob = priors_df["sp_prior_prob"].to_numpy(dtype=np.float32)
    live_suppression = priors_df["live_play_suppression"].to_numpy(dtype=np.float32)
    live_onset_sig = priors_df["live_tag_onset_signal"].to_numpy(dtype=np.float32)
    restart_hint = priors_df["restart_team_hint"].to_numpy()
    sp_type_arr = priors_df["sp_type"].to_numpy()

    kin_score = kin_df["kinematic_d_score"].to_numpy(dtype=np.float32)
    ball_oob = kin_df["ball_out_of_bounds"].to_numpy(dtype=bool)
    death_flux = kin_df["ball_death_flux"].to_numpy(dtype=np.float32)
    bx_centered = kin_df["ball_x_centered"].to_numpy(dtype=np.float32)
    by_centered = kin_df["ball_y_centered"].to_numpy(dtype=np.float32)

    ball_status_code = stage1_df["ball_status_code"].to_numpy(dtype=np.int8)
    ball_visible = ball_status_code == 0
    if "brd_valid" in stage1_df.columns and "brd_ball_x" in stage1_df.columns:
        ball_visible = ball_visible | (
            (stage1_df["brd_valid"].to_numpy(dtype=np.int8) == 1)
            & (~stage1_df["brd_ball_x"].isna().to_numpy())
        )

    p_in_window = 0.48 * sp_prior_prob + 0.52 * kin_score
    oob_confident = ball_oob & (ball_visible | (kin_score > 0.48))
    p_out_window = np.where(
        oob_confident,
        np.clip(0.58 + 0.38 * kin_score, 0.0, 0.96),
        np.clip((kin_score - 0.30) * 0.95 - 0.75 * live_suppression, 0.0, 0.78),
    )

    p_dead_raw = np.where(is_sp_window, p_in_window, p_out_window).astype(np.float32)

    if "frames_since_brd_replay" in stage1_df.columns:
        fs_rep = stage1_df["frames_since_brd_replay"].to_numpy(dtype=np.int32)
        just_after_replay = (fs_rep > 0) & (fs_rep <= 60)
        live_dynamics = (
            (stage1_df["all_players_mean_speed"].to_numpy(dtype=np.float32) > 2.55)
            | (stage1_df["ball_speed"].to_numpy(dtype=np.float32) > 4.5)
        )
        p_dead_raw = np.where(just_after_replay & live_dynamics, p_dead_raw * 0.18, p_dead_raw)

    if "gk_holding_ball_proxy" in stage1_df.columns:
        gk_hold = stage1_df["gk_holding_ball_proxy"].to_numpy(dtype=np.int8) != 0
        p_dead_raw = np.where(gk_hold & ~is_sp_window, p_dead_raw * 0.22, p_dead_raw)

    restart_flux = _compute_restart_impulse_signal(stage1_df, restart_hint, live_onset_sig)
    p_dead_snapped = _snap_both_deadball_edges(
        p_dead_raw=p_dead_raw,
        is_sp_window=is_sp_window,
        sp_type_arr=sp_type_arr,
        death_flux=death_flux,
        restart_flux=restart_flux,
        kinematic_d_score=kin_score,
        ball_x_centered=bx_centered,
        ball_y_centered=by_centered,
    )

    period_arr = (
        stage1_df["period"].to_numpy(dtype=np.int8)
        if "period" in stage1_df.columns
        else np.ones(N_FRAMES, dtype=np.int8)
    )
    p_dead = _viterbi_and_posterior_decode(
        p_dead_snapped=p_dead_snapped,
        death_flux=death_flux,
        restart_flux=restart_flux,
        period_arr=period_arr,
        base_switch_penalty=8.5,
        min_live_gap_frames=22,
        min_dead_duration_frames=25,
    )

    if "period" in stage1_df.columns:
        non_play_period = period_arr == 0
        p_dead = np.where(non_play_period, 1.0, p_dead)

    if "is_pre_kickoff_d" in stage1_df.columns:
        pre_ko = stage1_df["is_pre_kickoff_d"].to_numpy(dtype=np.int8) == 1
        p_dead = np.where(pre_ko, 1.0, p_dead)

    if "st_post_goal_dead" in stage1_df.columns:
        post_goal = stage1_df["st_post_goal_dead"].to_numpy(dtype=np.int8) == 1
        p_dead = np.where(post_goal, 0.99, p_dead)

    p_dead = np.nan_to_num(p_dead, nan=0.0, posinf=1.0, neginf=0.0)
    p_dead = np.clip(p_dead, 0.0, 1.0).astype(np.float32)

    missing_hint_mask = pd.isna(restart_hint) & (p_dead >= 0.50)
    if np.any(missing_hint_mask) and "near1_dist_diff_m" in stage1_df.columns:
        team_signal = stage1_df["near1_dist_diff_m"].to_numpy(dtype=np.float32).copy()
        if "near1_diff_leadlag_25f" in stage1_df.columns:
            team_signal += 0.5 * stage1_df["near1_diff_leadlag_25f"].to_numpy(dtype=np.float32)
        if "pitch_ctrl_leadlag_25f" in stage1_df.columns:
            team_signal += 4.5 * stage1_df["pitch_ctrl_leadlag_25f"].to_numpy(dtype=np.float32)

        restart_hint = restart_hint.astype(object)
        restart_hint[missing_hint_mask & (team_signal >= 0.0)] = "H"
        restart_hint[missing_hint_mask & (team_signal < 0.0)] = "A"

    result_df = pd.DataFrame({
        "frame_id": np.arange(N_FRAMES, dtype=np.int32),
        "p_dead": p_dead,
        "is_set_piece_window": is_sp_window.astype(bool),
        "restart_team_hint": pd.Categorical(restart_hint, categories=["H", "A"]),
    })

    assert len(result_df) == N_FRAMES, f"Invalid row count: {len(result_df)} != {N_FRAMES}"
    assert not result_df["frame_id"].isna().any(), "Column 'frame_id' contains NaN values!"
    assert not result_df["p_dead"].isna().any(), "Column 'p_dead' contains NaN values!"
    assert not result_df["is_set_piece_window"].isna().any(), "Column 'is_set_piece_window' contains NaN!"
    assert (result_df["frame_id"].to_numpy() == np.arange(N_FRAMES)).all(), "Frame ordering violation 0..199318!"

    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)

    if public_gt_path is not None:
        diag = evaluate_deadball_on_public_gt(result_df, public_gt_path=public_gt_path)
        if diag is not None:
            print(
                f"[Sanity-Check 5-min Public GT] D_F1={diag['d_f1']:.4f} "
                f"(Prec={diag['d_precision']:.4f}, Rec={diag['d_recall']:.4f}) | "
                f"D Frames: pred={diag['pred_d_frames']} vs gt={diag['true_d_frames']} | "
                f"D Segments: pred={diag['pred_d_segments']} vs gt={diag['true_d_segments']}"
            )

    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate stage2_deadball.parquet (Role 3 Dead-Ball Model).")
    parser.add_argument(
        "--stage1",
        type=str,
        default="data/interim/stage1_tracking.parquet",
        help="Path to stage1_tracking.parquet",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default="data/tactical/BAR-ATH_tactical_smart_tagging.xml",
        help="Path to BAR-ATH_tactical_smart_tagging.xml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/interim/stage2_deadball.parquet",
        help="Output path for stage2_deadball.parquet",
    )
    parser.add_argument(
        "--check-public-gt",
        action="store_true",
        help="Run read-only sanity check on the 5-minute public GT slice",
    )
    parser.add_argument(
        "--public-gt",
        type=str,
        default="evaluation/gt/BAR-ATH_possession_gt_public.csv",
        help="Path to public 5-minute GT CSV",
    )
    args = parser.parse_args()

    xml_file = args.xml if Path(args.xml).exists() else None
    gt_file = args.public_gt if args.check_public_gt else None

    df_out = run_deadball_pipeline(
        stage1_parquet_path=args.stage1,
        smart_tags_xml_path=xml_file,
        output_parquet_path=args.output,
        public_gt_path=gt_file,
    )
    print(f"Successfully saved {len(df_out)} frames to {args.output}.")