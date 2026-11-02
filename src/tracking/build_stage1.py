"""
Role 2 (Tracking): Generating the data contract data/interim/stage1_tracking.parquet (v3.5 Top-1% Pure Tactical).

Upgrades in v3.3 (Top-1% Edition):
  1. Fixed Tracklet Healer: removed the global `used_slots` lock that limited each slot to 1 stitch per match;
     replaced with chronological segment-level conflict verification.
  2. Full Smart Tagging Decomposition & Occlusion-Safe Sideline Resolver:
     - Explicit flags for Corner Kick, Goal Kick, Free Kick, Kick Off, and Goals + post-goal dead-ball window.
     - Throw-in N/A resolver uses the `Side` tag (Left/Right) and sideline proximity when the ball is occluded.
  3. Goalkeeper Proxy & Box Compression: identifies the deepest defender/GK per frame, computing distance
     to ball (`home_gk_dist_m`, `away_gk_dist_m`) and `gk_in_possession_proxy`.
  4. Dead-Ball & Transition Heuristics: `all_players_mean_speed`, `players_walking_ratio`,
     `ball_stationary_frames`, and `collective_vx_norm` (synchronized formation shift).
  5. Homography & Camera Dynamics: extracts `homography_reproj_err`, `camera_pan_speed`, and projects
     ball coordinates into pixel space (`ball_in_camera_fov`).

Upgrades in v3.4 (Optimization focused):
  1. Stripped all Pandas rolling windows inside loops and replaced them with ultra-fast SciPy C-backend
     `convolve1d` (for centered offline windows) and `lfilter` (for causal online windows).
  2. Optimized the Smart Tagging N/A Resolver: reuses pre-computed global `near1_dist_m` arrays instead of
     re-computing Euclidean distances player-by-player for every single XML restart instance.
  3. Optimized Homography loading: swapped slow Pandas .loc masking with raw NumPy contiguous memory views.
  4. Vectorized the Touch/Control State Machine: replaced python for-loop iteration with fast event-driven arrays.

Upgrades in v3.5 (Optimization focused):
  1. Refined stationary ball logic to use pure causal `lfilter`.
  2. Enhanced Smart Tagging N/A Resolver: Throw-in logic correctly relies on proximity to sidelines (y=0 / y=PITCH_WIDTH).
  3. Optimized Tracklet Healer: 100% Vectorized the Hungarian cost matrix construction directly in NumPy.
  4. Code clean-up: Fixed encoding, unified docstrings, and strictly enforced English comments.
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_coeffs, lfilter

# ---------------------------------------------------------
# 1. REPOSITORY ROOT & GAMES.PY INTEGRATION
# ---------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if not os.path.exists(os.path.join(ROOT, "games.py")) and os.path.exists(os.path.join(os.getcwd(), "games.py")):
    ROOT = os.path.abspath(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import games
    GAME = getattr(games, "GAME", "BAR-ATH")
    PITCH_LENGTH, PITCH_WIDTH = games.GAMES[GAME]["pitch"]
    FPS = float(games.GAMES[GAME]["fps"])
except Exception:
    games = None
    GAME = "BAR-ATH"
    PITCH_LENGTH, PITCH_WIDTH = 105.3, 68.0
    FPS = 25.0

DT = 1.0 / FPS

# Domain-grounded physical constants
SHORT_GAP_MAX_FRAMES = 25
CONTESTED_RADIUS_M = 2.5
CONTROL_RADIUS_3M = 3.0
SUPPORT_RADIUS_5M = 5.0
CLONE_DIST_THRESHOLD_M = 0.45
WINDOW_2S_FRAMES = 50
WINDOW_4S_FRAMES = 100
REACTION_TIME_S = 0.40
PLAYER_MAX_SPEED_MS = 7.50
PITCH_CONTROL_TAU_S = 0.50
MAX_BALL_SPEED_MS = 38.0
SAVGOL_WINDOW = 7
SAVGOL_POLY = 2

# Official frame boundaries
H1_START, H1_END = 11185, 90651
H2_START, H2_END = 110808, 190197
KICKOFF_2ND_HALF = 113742

HOME, AWAY = 0, 1


def resolve_match_paths():
    base_dir = os.environ.get("MTAH_DATA_DIR", os.path.join(ROOT, "data"))
    tac_dir = os.path.join(base_dir, "tactical")

    if games is not None and hasattr(games, "asset_paths"):
        t_paths = games.asset_paths(game=GAME)
        meta_path = t_paths["atd_meta"]
        raw_path = t_paths["atd_raw"]
        st_path = t_paths.get("smart_tagging", os.path.join(tac_dir, f"{GAME}_tactical_smart_tagging.xml"))
        homo_path = t_paths.get("homography", os.path.join(tac_dir, f"{GAME}_tactical_homography.csv"))
    else:
        meta_path = os.path.join(tac_dir, f"{GAME}_tactical_atd.xml")
        raw_path = os.path.join(tac_dir, f"{GAME}_tactical_atd.txt")
        st_path = os.path.join(tac_dir, f"{GAME}_tactical_smart_tagging.xml")
        homo_path = os.path.join(tac_dir, f"{GAME}_tactical_homography.csv")

    if games is not None and hasattr(games, "half_video_blocks"):
        blocks = games.half_video_blocks(GAME)
        (_, h1_s, h1_e), (_, h2_s, h2_e) = blocks[0], blocks[1]
    else:
        h1_s, h1_e, h2_s, h2_e = H1_START, H1_END, H2_START, H2_END

    n_frames = int(games.source_frames("tactical", GAME)) if games is not None and hasattr(games, "source_frames") else 199319
    return meta_path, raw_path, st_path, homo_path, n_frames, (h1_s, h1_e, h2_s, h2_e)


def get_slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_info = {}
    for team in root.iter("Team"):
        tid = team.get("id") or team.get("Id") or ""
        tname = team.get("name") or team.get("Name") or (team.text or "").strip()
        for child in team:
            if "name" in child.tag.lower() and child.text: tname = child.text.strip()
        team_info[tid] = tname

    home_id = None
    for tid, tname in team_info.items():
        if "bar" in tname.lower() or "fcb" in tname.lower() or "home" in tname.lower():
            home_id = tid
            break
    if home_id is None: home_id = list(team_info.keys())[0]

    players = list(root.iter("Player"))
    n_slots = len(players)
    slot_teams = np.full(n_slots, -1, dtype=np.int8)

    for idx, player in enumerate(players):
        tid = player.get("teamId") or player.get("TeamId")
        shirt_str = player.get("shirtNumber") or player.get("ShirtNumber")
        if shirt_str is None:
            for child in player:
                if "shirtnumber" in child.tag.lower() and child.text: shirt_str = child.text.strip()
        try:
            slot_idx = int(shirt_str) - 1 if shirt_str is not None else idx
            slot_idx = slot_idx if 0 <= slot_idx < n_slots else idx
        except ValueError:
            slot_idx = idx
        slot_teams[slot_idx] = HOME if tid == home_id else AWAY

    if np.any(slot_teams < 0):
        slot_teams = np.array([HOME if (p.get("teamId") or p.get("TeamId")) == home_id else AWAY for p in players], dtype=np.int8)
    return slot_teams


def parse_raw_atd(raw_path, n_frames, n_slots):
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    plen, pwid = PITCH_LENGTH, PITCH_WIDTH

    with open(raw_path, "r") as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3: continue
            frame_idx = int(parts[0]) - 1
            if not (0 <= frame_idx < n_frames): continue

            for slot, group in enumerate(parts[1].split(";")):
                if slot >= n_slots or not group or group[0] == "N": continue
                c1, c2 = group.find(","), group.find(",", group.find(",") + 1)
                px[frame_idx, slot] = float(group[:c1]) * plen
                py[frame_idx, slot] = float(group[c1 + 1:c2]) * pwid
                s_str = group[c2 + 1:]
                if s_str and s_str[0] != "N": ps[frame_idx, slot] = float(s_str)

            ball = parts[2]
            if ball and ball[0] != "N":
                c1, c2 = ball.find(","), ball.find(",", ball.find(",") + 1)
                bx[frame_idx] = float(ball[:c1]) * plen
                by[frame_idx] = float(ball[c1 + 1:c2]) * pwid

    speed_missing = ~np.isnan(px) & np.isnan(ps)
    if speed_missing.any():
        kin_speed = np.hypot(np.diff(px, axis=0, prepend=px[:1]), np.diff(py, axis=0, prepend=py[:1])) / DT
        ps[speed_missing] = np.where(np.isnan(kin_speed[speed_missing]), 0.0, np.minimum(kin_speed[speed_missing], 12.0))

    return px, py, ps, bx, by


def heal_player_tracks(px, py, ps, slot_teams, max_stitch_frames=25):
    n_frames, _ = px.shape
    px_h, py_h, ps_h = px.copy(), py.copy(), ps.copy()
    total_stitched, total_clones_removed = 0, 0

    for team_code in (HOME, AWAY):
        team_slots = np.flatnonzero(slot_teams == team_code)
        deaths, births = [], []

        for s in team_slots:
            valid = ~np.isnan(px_h[:, s])
            if valid.sum() < 15: continue
            idx = np.flatnonzero(valid)
            splits = np.flatnonzero(np.diff(idx) > 1)
            seg_starts = np.concatenate(([idx[0]], idx[splits + 1]))
            seg_ends = np.concatenate((idx[splits], [idx[-1]]))

            for start, end in zip(seg_starts, seg_ends):
                if start > 0 and (end - start) >= 10:
                    births.append((s, int(start), int(end), float(px_h[start, s]), float(py_h[start, s])))
                if 5 <= end < n_frames - 1 and (end - start) >= 10:
                    vx = float(px_h[end, s] - px_h[end - 4, s]) / (4.0 * DT)
                    vy = float(py_h[end, s] - py_h[end - 4, s]) / (4.0 * DT)
                    deaths.append((s, int(end), float(px_h[end, s]), float(py_h[end, s]), vx, vy))

        if deaths and births:
            n_d, n_b = len(deaths), len(births)
            
            # Vectorized Cost Matrix Construction
            d_slots = np.array([d[0] for d in deaths])
            d_fs = np.array([d[1] for d in deaths])
            dxs = np.array([d[2] for d in deaths])
            dys = np.array([d[3] for d in deaths])
            dvxs = np.array([d[4] for d in deaths])
            dvys = np.array([d[5] for d in deaths])

            b_slots = np.array([b[0] for b in births])
            b_fs = np.array([b[1] for b in births])
            b_xs = np.array([b[3] for b in births])
            b_ys = np.array([b[4] for b in births])

            gap = b_fs[None, :] - d_fs[:, None]
            valid_gap = (gap > 0) & (gap <= max_stitch_frames)
            valid_slot = d_slots[:, None] != b_slots[None, :]
            valid_mask = valid_gap & valid_slot

            proj_x = dxs[:, None] + dvxs[:, None] * gap * DT
            proj_y = dys[:, None] + dvys[:, None] * gap * DT

            dist = np.hypot(b_xs[None, :] - proj_x, b_ys[None, :] - proj_y)
            cost_mat = np.where(valid_mask & (dist <= 2.5), dist, 1e6).astype(np.float32)

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            valid_pairs = sorted([(deaths[r][1], r, c) for r, c in zip(row_ind, col_ind) if cost_mat[r, c] < 2.5], key=lambda x: x[0])
            slot_redirect = {}

            for _, r, c in valid_pairs:
                orig_d_slot, d_f, dx, dy, dvx, dvy = deaths[r]
                b_slot, b_f, b_end, bx_val, by_val = births[c]
                d_slot = slot_redirect.get((orig_d_slot, d_f), orig_d_slot)
                
                if d_slot == b_slot or np.any(np.isnan(px_h[b_f:b_end + 1, b_slot])) or not np.all(np.isnan(px_h[d_f + 1:b_end + 1, d_slot])):
                    continue

                alpha = np.linspace(0.0, 1.0, b_f - d_f + 1, dtype=np.float32)[1:-1]
                if len(alpha) > 0:
                    px_h[d_f + 1:b_f, d_slot] = (1.0 - alpha) * dx + alpha * bx_val
                    py_h[d_f + 1:b_f, d_slot] = (1.0 - alpha) * dy + alpha * by_val
                    ps_h[d_f + 1:b_f, d_slot] = (1.0 - alpha) * np.hypot(dvx, dvy) + alpha * ps_h[b_f, b_slot]

                seg_slice = slice(b_f, b_end + 1)
                px_h[seg_slice, d_slot], py_h[seg_slice, d_slot], ps_h[seg_slice, d_slot] = px_h[seg_slice, b_slot], py_h[seg_slice, b_slot], ps_h[seg_slice, b_slot]
                px_h[seg_slice, b_slot], py_h[seg_slice, b_slot], ps_h[seg_slice, b_slot] = np.nan, np.nan, np.nan
                slot_redirect[(b_slot, b_end)] = d_slot
                total_stitched += 1

        tx, ty = px_h[:, team_slots], py_h[:, team_slots]
        valid_team = (~np.isnan(tx)).astype(np.float32)
        stability_score = convolve1d(valid_team, np.ones(51, dtype=np.float32), axis=0, mode="constant", cval=0.0) + (valid_team.sum(axis=0)[None, :] * 1e-6)

        m_team = len(team_slots)
        for i in range(m_team):
            for j in range(i + 1, m_team):
                clones = np.hypot(tx[:, i] - tx[:, j], ty[:, i] - ty[:, j]) < CLONE_DIST_THRESHOLD_M
                if not clones.any(): continue
                drop_j, drop_i = clones & (stability_score[:, i] >= stability_score[:, j]), clones & (stability_score[:, i] < stability_score[:, j])
                
                if drop_j.any():
                    s_j = team_slots[j]
                    px_h[drop_j, s_j], py_h[drop_j, s_j], ps_h[drop_j, s_j] = np.nan, np.nan, np.nan
                    tx[drop_j, j], ty[drop_j, j] = np.nan, np.nan
                    total_clones_removed += int(drop_j.sum())
                if drop_i.any():
                    s_i = team_slots[i]
                    px_h[drop_i, s_i], py_h[drop_i, s_i], ps_h[drop_i, s_i] = np.nan, np.nan, np.nan
                    tx[drop_i, i], ty[drop_i, i] = np.nan, np.nan
                    total_clones_removed += int(drop_i.sum())

    print(f"[TRACKLET HEALER] Stitched {total_stitched} tracklets; removed {total_clones_removed} clone observations.")
    return px_h, py_h, ps_h


def compute_player_velocities_savgol(px, py, ps):
    valid_mask = ~np.isnan(px)
    px_z, py_z = np.where(valid_mask, px, 0.0).astype(np.float32), np.where(valid_mask, py, 0.0).astype(np.float32)
    c_pos = savgol_coeffs(SAVGOL_WINDOW, SAVGOL_POLY, deriv=0, use="conv").astype(np.float32)
    c_vel = (savgol_coeffs(SAVGOL_WINDOW, SAVGOL_POLY, deriv=1, use="conv") / DT).astype(np.float32)

    has_7 = convolve1d(valid_mask.astype(np.float32), np.ones(SAVGOL_WINDOW, dtype=np.float32), axis=0, mode="constant", cval=0.0) > (SAVGOL_WINDOW - 0.5)

    px_sg = convolve1d(px_z, c_pos, axis=0, mode="nearest")
    py_sg = convolve1d(py_z, c_pos, axis=0, mode="nearest")
    px_smooth = np.where(has_7, px_sg, px).astype(np.float32)
    py_smooth = np.where(has_7, py_sg, py).astype(np.float32)

    vx_sg = convolve1d(px_z, c_vel, axis=0, mode="nearest")
    vy_sg = convolve1d(py_z, c_vel, axis=0, mode="nearest")

    dx_f, dy_f = np.diff(px_smooth, axis=0, append=np.nan), np.diff(py_smooth, axis=0, append=np.nan)
    dx_b, dy_b = np.diff(px_smooth, axis=0, prepend=np.nan), np.diff(py_smooth, axis=0, prepend=np.nan)
    has_f, has_b = ~np.isnan(dx_f), ~np.isnan(dx_b)

    vx_fb = np.where(has_f & has_b, 0.5 * (dx_f + dx_b) / DT, np.where(has_f, dx_f / DT, np.where(has_b, dx_b / DT, 0.0)))
    vy_fb = np.where(has_f & has_b, 0.5 * (dy_f + dy_b) / DT, np.where(has_f, dy_f / DT, np.where(has_b, dy_b / DT, 0.0)))

    vx, vy = np.where(has_7, vx_sg, vx_fb).astype(np.float32), np.where(has_7, vy_sg, vy_fb).astype(np.float32)
    spd_kin = np.hypot(vx, vy)
    unphys = (spd_kin > 12.5) | (~valid_mask)
    vx[unphys], vy[unphys] = 0.0, 0.0

    ps_smooth = np.where(valid_mask, 0.5 * np.nan_to_num(ps, nan=0.0) + 0.5 * np.minimum(spd_kin, 12.0), np.nan).astype(np.float32)
    return px_smooth, py_smooth, ps_smooth, vx, vy


def compute_focal_attention_centroid(px, py, ps, pvx, pvy):
    valid = ~np.isnan(px)
    n_visible = valid.sum(axis=1)
    px_z, py_z = np.where(valid, px, 0.0), np.where(valid, py, 0.0)

    spd = np.maximum(0.0, np.nan_to_num(ps, nan=0.0))
    w_coarse = np.where(valid, 1.0 + np.log1p(spd), 0.0).astype(np.float32)
    w_sum1 = np.maximum(1e-5, w_coarse.sum(axis=1))
    cx1 = np.where(n_visible > 0, (px_z * w_coarse).sum(axis=1) / w_sum1, PITCH_LENGTH / 2.0)
    cy1 = np.where(n_visible > 0, (py_z * w_coarse).sum(axis=1) / w_sum1, PITCH_WIDTH / 2.0)

    dx1, dy1 = np.where(valid, cx1[:, None] - px_z, 0.0), np.where(valid, cy1[:, None] - py_z, 0.0)
    dist1 = np.hypot(dx1, dy1)
    
    conv_speed = np.maximum(0.0, (pvx * dx1 + pvy * dy1) / (dist1 + 0.5))
    spatial_kernel = np.exp(-0.5 * (dist1 / 18.0) ** 2)
    w_focal = np.where(valid, w_coarse * (0.35 + 0.65 * spatial_kernel) * (1.0 + 0.25 * conv_speed), 0.0).astype(np.float32)

    w_sum2 = np.maximum(1e-5, w_focal.sum(axis=1))
    att_x = np.where(n_visible > 0, (px_z * w_focal).sum(axis=1) / w_sum2, PITCH_LENGTH / 2.0).astype(np.float32)
    att_y = np.where(n_visible > 0, (py_z * w_focal).sum(axis=1) / w_sum2, PITCH_WIDTH / 2.0).astype(np.float32)
    spread_m = np.sqrt(np.where(n_visible > 0, (((px_z - att_x[:, None]) ** 2 + (py_z - att_y[:, None]) ** 2) * w_focal).sum(axis=1) / w_sum2, 100.0)).astype(np.float32)
    
    return att_x, att_y, spread_m


def compute_gap_boundary_timers(status_code):
    n_frames = len(status_code)
    is_long_gap = (status_code == 2)
    gap_indices = np.flatnonzero(is_long_gap)
    frames_since = np.full(n_frames, 10000, dtype=np.int32)
    frames_to = np.full(n_frames, 10000, dtype=np.int32)
    
    if len(gap_indices) == 0: return frames_since, frames_to
    idx = np.arange(n_frames)
    
    right_idx = np.searchsorted(gap_indices, idx, side="left")
    valid_right = right_idx < len(gap_indices)
    frames_to[valid_right] = np.minimum(10000, gap_indices[right_idx[valid_right]] - idx[valid_right])
    frames_to[is_long_gap] = 0

    left_idx = np.searchsorted(gap_indices, idx, side="right") - 1
    valid_left = left_idx >= 0
    frames_since[valid_left] = np.minimum(10000, idx[valid_left] - gap_indices[left_idx[valid_left]])
    frames_since[is_long_gap] = 0

    return frames_since, frames_to


def impute_ball_and_uncertainty(bx, by, att_x, att_y, att_spread):
    n_frames = len(bx)
    observed = ~np.isnan(bx)
    ball_x, ball_y = bx.copy(), by.copy()
    status_code = np.full(n_frames, 2, dtype=np.int8)
    status_code[observed] = 0
    sigma_m = np.full(n_frames, 8.0, dtype=np.float32)
    sigma_m[observed] = 0.5
    gap_length_arr = np.zeros(n_frames, dtype=np.int32)

    padded = np.concatenate(([True], observed, [True]))
    gap_starts, gap_ends = np.flatnonzero(padded[:-1] & ~padded[1:]), np.flatnonzero(~padded[:-1] & padded[1:])

    for start, end in zip(gap_starts, gap_ends):
        gap_len = end - start
        gap_length_arr[start:end] = gap_len
        has_left, has_right = start > 0, end < n_frames

        if gap_len <= SHORT_GAP_MAX_FRAMES and has_left and has_right:
            status_code[start:end] = 1
            alpha = np.linspace(0.0, 1.0, gap_len + 2, dtype=np.float32)[1:-1]
            ball_x[start:end] = (1.0 - alpha) * bx[start - 1] + alpha * bx[end]
            ball_y[start:end] = (1.0 - alpha) * by[start - 1] + alpha * by[end]
            sigma_m[start:end] = 0.5 + 0.16 * np.minimum(np.arange(1, gap_len + 1), np.arange(gap_len, 0, -1))
        else:
            status_code[start:end] = 2
            est_x, est_y = att_x[start:end].copy(), att_y[start:end].copy()
            dist_to_edge = np.minimum(np.arange(1, gap_len + 1), np.arange(gap_len, 0, -1)).astype(np.float32)

            if has_left and has_right:
                alpha = np.linspace(0.0, 1.0, gap_len + 2, dtype=np.float32)[1:-1]
                w_bridge = np.exp(-dist_to_edge / 40.0).astype(np.float32)
                est_x = w_bridge * ((1.0 - alpha) * bx[start - 1] + alpha * bx[end]) + (1.0 - w_bridge) * est_x
                est_y = w_bridge * ((1.0 - alpha) * by[start - 1] + alpha * by[end]) + (1.0 - w_bridge) * est_y
            else:
                blend_len = min(25, gap_len)
                if has_left:
                    w = np.linspace(0.85, 0.0, blend_len, dtype=np.float32)
                    est_x[:blend_len] = w * bx[start - 1] + (1.0 - w) * est_x[:blend_len]
                    est_y[:blend_len] = w * by[start - 1] + (1.0 - w) * est_y[:blend_len]
                if has_right:
                    w = np.linspace(0.0, 0.85, blend_len, dtype=np.float32)
                    est_x[-blend_len:] = w * bx[end] + (1.0 - w) * est_x[-blend_len:]
                    est_y[-blend_len:] = w * by[end] + (1.0 - w) * est_y[-blend_len:]

            ball_x[start:end], ball_y[start:end] = est_x, est_y
            sigma_m[start:end] = np.clip(0.32 * att_spread[start:end] * (1.0 - 0.5 * np.exp(-dist_to_edge / 35.0)), 2.5, 25.0)

    status_label = np.array(["OBSERVED", "INTERPOLATED_SHORT", "INFERRED_LATENT"])[status_code]
    frames_since_gap, frames_to_gap = compute_gap_boundary_timers(status_code)
    out_of_bounds = ((ball_x < 0.5) | (ball_x > PITCH_LENGTH - 0.5) | (ball_y < 0.5) | (ball_y > PITCH_WIDTH - 0.5)).astype(np.int8)

    return ball_x, ball_y, status_code, status_label, sigma_m, gap_length_arr, frames_since_gap, frames_to_gap, out_of_bounds


def compute_clean_ball_kinematics(ball_x, ball_y, status_code):
    is_obs = (status_code <= 1).astype(np.float32)
    same_5 = (convolve1d(is_obs, np.ones(5, dtype=np.float32), axis=0, mode="constant", cval=0.0) > 4.5) | \
             (convolve1d(is_obs, np.ones(5, dtype=np.float32), axis=0, mode="constant", cval=0.0) < 0.5)

    c_vel5 = (savgol_coeffs(5, 2, deriv=1, use="conv") / DT).astype(np.float32)
    vx_sg = convolve1d(ball_x, c_vel5, axis=0, mode="nearest")
    vy_sg = convolve1d(ball_y, c_vel5, axis=0, mode="nearest")

    same_next = (status_code[:-1] <= 1) == (status_code[1:] <= 1)
    has_f, has_b = np.concatenate((same_next, [False])), np.concatenate(([False], same_next))

    dx_f, dy_f = np.diff(ball_x, append=ball_x[-1]), np.diff(ball_y, append=ball_y[-1])
    dx_b, dy_b = np.diff(ball_x, prepend=ball_x[0]), np.diff(ball_y, prepend=ball_y[0])

    vx_fb = np.where(has_f & has_b, 0.5 * (dx_f + dx_b) / DT, np.where(has_f, dx_f / DT, np.where(has_b, dx_b / DT, 0.0)))
    vy_fb = np.where(has_f & has_b, 0.5 * (dy_f + dy_b) / DT, np.where(has_f, dy_f / DT, np.where(has_b, dy_b / DT, 0.0)))

    vx = np.where(same_5, vx_sg, vx_fb).astype(np.float32)
    vy = np.where(same_5, vy_sg, vy_fb).astype(np.float32)

    speed = np.hypot(vx, vy).astype(np.float32)
    over_max = speed > MAX_BALL_SPEED_MS
    if over_max.any():
        scale = MAX_BALL_SPEED_MS / np.maximum(speed, 1e-5)
        vx = np.where(over_max, vx * scale, vx).astype(np.float32)
        vy = np.where(over_max, vy * scale, vy).astype(np.float32)
        speed = np.minimum(speed, MAX_BALL_SPEED_MS).astype(np.float32)

    ax = np.where(has_f & has_b, 0.5 * (np.diff(vx, append=vx[-1]) + np.diff(vx, prepend=vx[0])) / DT, np.where(has_f, np.diff(vx, append=vx[-1]) / DT, 0.0))
    ay = np.where(has_f & has_b, 0.5 * (np.diff(vy, append=vy[-1]) + np.diff(vy, prepend=vy[0])) / DT, np.where(has_f, np.diff(vy, append=vy[-1]) / DT, 0.0))
    acc = np.minimum(np.hypot(ax, ay), 150.0).astype(np.float32)

    return vx, vy, speed, acc


# ---------------------------------------------------------
# 4. SMART TAGGING FULL DECOMPOSITION & OPTIMIZED N/A RESOLVER
# ---------------------------------------------------------
def resolve_smart_tagging_na_and_enhance(st_path, n_frames, h_near1, a_near1, home_cy, away_cy, ball_speed, status_code):
    st_set_piece = np.zeros(n_frames, dtype=np.int8)
    st_throw_in = np.zeros(n_frames, dtype=np.int8)
    st_corner_kick = np.zeros(n_frames, dtype=np.int8)
    st_goal_kick = np.zeros(n_frames, dtype=np.int8)
    st_free_kick = np.zeros(n_frames, dtype=np.int8)
    st_kick_off = np.zeros(n_frames, dtype=np.int8)
    st_goal = np.zeros(n_frames, dtype=np.int8)
    st_post_goal_dead = np.zeros(n_frames, dtype=np.int8)
    st_defending_tag = np.zeros(n_frames, dtype=np.int8)
    st_in_box_tag = np.zeros(n_frames, dtype=np.int8)
    st_restart_team = np.zeros(n_frames, dtype=np.int8)
    home_votes = np.zeros(n_frames, dtype=np.float32)
    away_votes = np.zeros(n_frames, dtype=np.float32)
    st_n_active = np.zeros(n_frames, dtype=np.int8)

    if not os.path.exists(st_path):
        return {
            "st_set_piece": st_set_piece, "st_throw_in": st_throw_in, "st_corner_kick": st_corner_kick,
            "st_goal_kick": st_goal_kick, "st_free_kick": st_free_kick, "st_kick_off": st_kick_off,
            "st_goal": st_goal, "st_post_goal_dead": st_post_goal_dead, "st_defending_tag": st_defending_tag,
            "st_in_box_tag": st_in_box_tag, "st_restart_team": st_restart_team,
            "st_team_prior": np.zeros(n_frames, dtype=np.int8), "st_team_score": np.zeros(n_frames, dtype=np.float32),
            "st_n_active_tags": st_n_active,
        }

    resolved_na_count, inverted_def_count = 0, 0
    goal_end_frames, kickoff_after_goal_starts = [], []

    root = ET.parse(st_path).getroot()
    for inst in root.iter("instance"):
        start_el, end_el, code_el = inst.find("start"), inst.find("end"), inst.find("code")
        if start_el is None or end_el is None or code_el is None: continue

        s_frame = max(0, min(n_frames - 1, int(round(float(start_el.text) * FPS))))
        e_frame = max(0, min(n_frames, int(round(float(end_el.text) * FPS)) + 1))
        code = (code_el.text or "").strip().upper()

        tags = {label.find("group").text.strip().lower(): label.find("text").text.strip()
                for label in inst.findall("label") if label.find("group") is not None and label.find("text") is not None and label.find("text").text}
        
        team_val = tags.get("team", "N/A").lower()
        type_val = tags.get("type", "").lower()
        st_n_active[s_frame:e_frame] = np.minimum(120, st_n_active[s_frame:e_frame] + 1)

        is_defending = code.startswith("DEFEND")
        is_goal = (code == "GOALS")
        is_set_piece = (code == "SET PIECES")
        
        if is_defending: st_defending_tag[s_frame:e_frame] = 1
        if "IN THE BOX" in code: st_in_box_tag[s_frame:e_frame] = 1
        if is_goal:
            st_goal[s_frame:e_frame] = 1
            goal_end_frames.append(e_frame)
            
        if is_set_piece:
            st_set_piece[s_frame:e_frame] = 1
            if "throw-in" in type_val: st_throw_in[s_frame:e_frame] = 1
            elif "corner" in type_val: st_corner_kick[s_frame:e_frame] = 1
            elif "goal kick" in type_val: st_goal_kick[s_frame:e_frame] = 1
            elif "free kick" in type_val: st_free_kick[s_frame:e_frame] = 1
            elif "kick off" in type_val:
                st_kick_off[s_frame:e_frame] = 1
                if "after goal" in type_val: kickoff_after_goal_starts.append(s_frame)

        vote_end = min(n_frames, e_frame + 25)
        if "barcelona" in team_val:
            if is_defending:
                away_votes[s_frame:e_frame] += 1.0
                inverted_def_count += 1
            elif not is_goal:
                home_votes[s_frame:e_frame] += 1.0
                if is_set_piece: st_restart_team[max(0, e_frame - 10):vote_end] = 1
        elif "atletic" in team_val or "athletic" in team_val:
            if is_defending:
                home_votes[s_frame:e_frame] += 1.0
                inverted_def_count += 1
            elif not is_goal:
                away_votes[s_frame:e_frame] += 1.0
                if is_set_piece: st_restart_team[max(0, e_frame - 10):vote_end] = -1
        else:
            w_start, w_end = max(0, e_frame - 20), min(n_frames, e_frame + 20)
            if w_end > w_start:
                if "throw-in" in type_val:
                    h_y_med = float(np.nanmedian(home_cy[w_start:w_end]))
                    a_y_med = float(np.nanmedian(away_cy[w_start:w_end]))
                    h_side_dist = min(h_y_med, PITCH_WIDTH - h_y_med)
                    a_side_dist = min(a_y_med, PITCH_WIDTH - a_y_med)
                    if h_side_dist <= a_side_dist:
                        home_votes[s_frame:vote_end] += 0.85
                        if is_set_piece: st_restart_team[max(0, e_frame - 10):vote_end] = 1
                    else:
                        away_votes[s_frame:vote_end] += 0.85
                        if is_set_piece: st_restart_team[max(0, e_frame - 10):vote_end] = -1
                else:
                    h_med = float(np.nanmedian(h_near1[w_start:w_end]))
                    a_med = float(np.nanmedian(a_near1[w_start:w_end]))
                    if h_med <= a_med:
                        home_votes[s_frame:vote_end] += 0.85
                        if is_set_piece: st_restart_team[max(0, e_frame - 10):vote_end] = 1
                    else:
                        away_votes[s_frame:vote_end] += 0.85
                        if is_set_piece: st_restart_team[max(0, e_frame - 10):vote_end] = -1
                resolved_na_count += 1

    for g_end in goal_end_frames:
        next_ko = [ko for ko in kickoff_after_goal_starts if 0 < (ko - g_end) <= 3000]
        ko_end = next_ko[0] if next_ko else min(n_frames, g_end + 1500)
        st_post_goal_dead[g_end:ko_end] = 1
        home_votes[g_end:ko_end], away_votes[g_end:ko_end] = 0.0, 0.0

    net_score = (home_votes - away_votes).astype(np.float32)
    print(f"[SMART TAGGING ENHANCER] Inverted polarity on {inverted_def_count} DEFEND* tags; resolved {resolved_na_count} 'Team=N/A' instances.")
    return {
        "st_set_piece": st_set_piece, "st_throw_in": st_throw_in, "st_corner_kick": st_corner_kick,
        "st_goal_kick": st_goal_kick, "st_free_kick": st_free_kick, "st_kick_off": st_kick_off,
        "st_goal": st_goal, "st_post_goal_dead": st_post_goal_dead, "st_defending_tag": st_defending_tag,
        "st_in_box_tag": st_in_box_tag, "st_restart_team": st_restart_team,
        "st_team_prior": np.sign(net_score).astype(np.int8), "st_team_score": net_score, "st_n_active_tags": st_n_active,
    }


# ---------------------------------------------------------
# 5. HOMOGRAPHY & CAMERA PANNING DYNAMICS (OPTIMIZED)
# ---------------------------------------------------------
def extract_homography_features(homo_path, n_frames, ball_x, ball_y):
    reproj_err = np.full(n_frames, 0.25, dtype=np.float32)
    cam_pan_speed = np.zeros(n_frames, dtype=np.float32)
    ball_in_fov = np.ones(n_frames, dtype=np.int8)

    if not os.path.exists(homo_path):
        return {"homography_reproj_err": reproj_err, "camera_pan_speed": cam_pan_speed, "ball_in_camera_fov": ball_in_fov}

    df_h = pd.read_csv(homo_path)
    h_vals = df_h[["frame", "reprojection_error", "h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]].values
    frames = h_vals[:, 0].astype(np.int32)
    valid_mask = (frames >= 0) & (frames < n_frames)
    
    H_mat = np.zeros((n_frames, 9), dtype=np.float32)
    valid_frames = frames[valid_mask]
    H_mat[valid_frames] = h_vals[valid_mask, 2:]
    reproj_err[valid_frames] = np.nan_to_num(h_vals[valid_mask, 1].astype(np.float32), nan=1.5)

    bx_norm, by_norm = (ball_x / PITCH_LENGTH).astype(np.float32), (ball_y / PITCH_WIDTH).astype(np.float32)
    denom = H_mat[:, 6] * bx_norm + H_mat[:, 7] * by_norm + H_mat[:, 8]
    safe_denom = np.where(np.abs(denom) > 1e-5, denom, 1.0)
    u_ball = (H_mat[:, 0] * bx_norm + H_mat[:, 1] * by_norm + H_mat[:, 2]) / safe_denom
    v_ball = (H_mat[:, 3] * bx_norm + H_mat[:, 4] * by_norm + H_mat[:, 5]) / safe_denom

    ball_in_fov = ((u_ball >= -60.0) & (u_ball <= 1980.0) & (v_ball >= -60.0) & (v_ball <= 1140.0)).astype(np.int8)

    denom_c = H_mat[:, 6] * 0.5 + H_mat[:, 7] * 0.5 + H_mat[:, 8]
    safe_denom_c = np.where(np.abs(denom_c) > 1e-5, denom_c, 1.0)
    u_c = (H_mat[:, 0] * 0.5 + H_mat[:, 1] * 0.5 + H_mat[:, 2]) / safe_denom_c
    v_c = (H_mat[:, 3] * 0.5 + H_mat[:, 4] * 0.5 + H_mat[:, 5]) / safe_denom_c
    cam_pan_speed = np.minimum(np.hypot(np.diff(u_c, prepend=u_c[0]), np.diff(v_c, prepend=v_c[0])) * FPS, 2000.0).astype(np.float32)

    return {"homography_reproj_err": reproj_err, "camera_pan_speed": cam_pan_speed, "ball_in_camera_fov": ball_in_fov}


# ---------------------------------------------------------
# 6. KINEMATICS, TTI, PITCH CONTROL, GK PROXY & DEAD-BALL HEURISTICS
# ---------------------------------------------------------
def compute_ball_deflection_angle(ball_x, ball_y, status_code):
    obs = (status_code <= 1)
    obs_prev2, obs_next2 = np.roll(obs, 2), np.roll(obs, -2)
    obs_prev2[:2], obs_next2[-2:] = False, False

    dx_in, dy_in = ball_x - np.roll(ball_x, 2), ball_y - np.roll(ball_y, 2)
    dx_out, dy_out = np.roll(ball_x, -2) - ball_x, np.roll(ball_y, -2) - ball_y

    norm_in, norm_out = np.hypot(dx_in, dy_in), np.hypot(dx_out, dy_out)
    valid_defl = obs & obs_prev2 & obs_next2 & (norm_in > 0.25) & (norm_out > 0.25)
    
    cos_sim = np.clip((dx_in * dx_out + dy_in * dy_out) / np.where(valid_defl, norm_in * norm_out, 1.0), -1.0, 1.0)
    return np.where(valid_defl, np.arccos(cos_sim), 0.0).astype(np.float32)


def compute_touch_and_control_proxies(home_d1, away_d1, defl_angle, status_code, period):
    """
    Implements CHALLENGE.md possession rules strictly per half:
      - Counts distinct controlled touch episodes (separated by ball flight > 6 frames or team change)
        so a single touch has count=1 while >=2 consecutive passes/touches reach count>=2.
    """
    n_frames = len(home_d1)
    obs = (status_code <= 1)
    h_close = obs & (home_d1 <= 1.8) & (away_d1 > 2.0)
    a_close = obs & (away_d1 <= 1.8) & (home_d1 > 2.0)

    h_sust = convolve1d(h_close.astype(np.float32), np.ones(8, dtype=np.float32), axis=0, mode="constant") >= 4.0
    a_sust = convolve1d(a_close.astype(np.float32), np.ones(8, dtype=np.float32), axis=0, mode="constant") >= 4.0

    h_valid = h_sust | (h_close & (defl_angle < 1.3))
    a_valid = a_sust | (a_close & (defl_angle < 1.3))
    sust_any = h_sust | a_sust

    touch_team = np.where(h_valid, 1, np.where(a_valid, -1, 0)).astype(np.int8)

    home_consec, away_consec = np.zeros(n_frames, dtype=np.int16), np.zeros(n_frames, dtype=np.int16)
    last_ctrl, frames_since = np.zeros(n_frames, dtype=np.int8), np.full(n_frames, 10000, dtype=np.int32)

    for p_val in (1, 2):
        p_idx = np.flatnonzero(period == p_val)
        if len(p_idx) == 0:
            continue

        s, e = int(p_idx[0]), int(p_idx[-1]) + 1
        t_half = touch_team[s:e]
        sust_half = sust_any[s:e]
        active_idx = np.flatnonzero(t_half != 0)
        if len(active_idx) == 0:
            continue

        teams_active = t_half[active_idx]
        sust_active = sust_half[active_idx]
        # A new distinct touch episode starts when team changes OR after a flight gap > 6 frames (0.24s)
        is_new_ep = np.concatenate(([True], (teams_active[1:] != teams_active[:-1]) | (np.diff(active_idx) > 6)))

        consec = np.zeros(len(active_idx), dtype=np.int16)
        ctrl_team_active = np.zeros(len(active_idx), dtype=np.int8)
        cur_team, run_val = 0, 0

        for i in range(len(active_idx)):
            tm = int(teams_active[i])
            if i == 0 or tm != int(teams_active[i - 1]):
                run_val = 1
            elif is_new_ep[i]:
                run_val = min(25, run_val + 1)
            consec[i] = run_val
            # Possession flips on sustained control (>=4 frames) OR >=2 consecutive touches by same team
            if sust_active[i] or run_val >= 2:
                cur_team = tm
            ctrl_team_active[i] = cur_team

        dense_ctrl = np.zeros(e - s, dtype=np.int8)
        dense_ctrl[active_idx] = ctrl_team_active
        
        # FIX PANDAS 2.1+: Explicit .replace(0, np.nan).ffill() instead of deprecated replace(0, method='ffill')
        df_team = pd.Series(dense_ctrl).replace(0, np.nan).ffill().fillna(0).to_numpy(dtype=np.int8)
        last_ctrl[s:e] = df_team

        nz = np.flatnonzero(dense_ctrl != 0)
        if len(nz) > 0:
            idx_arr = np.arange(e - s)
            left_idx = np.searchsorted(nz, idx_arr, side="right") - 1
            valid_left = left_idx >= 0
            since_arr = np.full(e - s, 10000, dtype=np.int32)
            since_arr[valid_left] = np.minimum(10000, idx_arr[valid_left] - nz[left_idx[valid_left]])
            frames_since[s:e] = since_arr

        h_c, a_c = np.zeros(e - s, dtype=np.int16), np.zeros(e - s, dtype=np.int16)
        h_c[active_idx[teams_active == 1]] = consec[teams_active == 1]
        a_c[active_idx[teams_active == -1]] = consec[teams_active == -1]

        # FIX PANDAS 2.1+: Explicit .replace(0, np.nan).ffill() instead of deprecated replace(0, method='ffill')
        home_consec[s:e] = np.where(df_team == 1, pd.Series(h_c).replace(0, np.nan).ffill().fillna(0).to_numpy(dtype=np.int16), 0)
        away_consec[s:e] = np.where(df_team == -1, pd.Series(a_c).replace(0, np.nan).ffill().fillna(0).to_numpy(dtype=np.int16), 0)

    return {
        "home_consecutive_touches": home_consec,
        "away_consecutive_touches": away_consec,
        "last_controlled_team": last_ctrl,
        "frames_since_last_control": frames_since,
    }

def extract_canonical_team_features(px, py, ps, pvx, pvy, slot_teams, ball_x, ball_y, ball_vx, ball_vy, ball_speed, status_code, period):
    feats = {}
    n_frames = px.shape[0]
    pitch_ctrl_rates = {}

    for team_code, prefix in ((HOME, "home"), (AWAY, "away")):
        mask = (slot_teams == team_code)
        tx, ty, ts = px[:, mask], py[:, mask], ps[:, mask]
        tvx, tvy = pvx[:, mask], pvy[:, mask]

        valid_pos, valid_spd = ~np.isnan(tx), ~np.isnan(ts)
        n_vis, n_spd = valid_pos.sum(axis=1).astype(np.int8), valid_spd.sum(axis=1)
        feats[f"{prefix}_n_visible"] = n_vis

        safe_n_vis, safe_n_spd = np.maximum(1, n_vis).astype(np.float32), np.maximum(1, n_spd).astype(np.float32)
        tx_z, ty_z, ts_z = np.where(valid_pos, tx, 0.0), np.where(valid_pos, ty, 0.0), np.where(valid_spd, ts, 0.0)

        cx = np.where(n_vis > 0, tx_z.sum(axis=1) / safe_n_vis, PITCH_LENGTH / 2.0).astype(np.float32)
        cy = np.where(n_vis > 0, ty_z.sum(axis=1) / safe_n_vis, PITCH_WIDTH / 2.0).astype(np.float32)
        feats[f"{prefix}_centroid_x"], feats[f"{prefix}_centroid_y"] = cx, cy
        feats[f"{prefix}_centroid_vx"] = np.gradient(cx, DT).astype(np.float32)
        feats[f"{prefix}_centroid_vy"] = np.gradient(cy, DT).astype(np.float32)
        feats[f"{prefix}_mean_speed"] = np.where(n_spd > 0, ts_z.sum(axis=1) / safe_n_spd, 0.0).astype(np.float32)

        var_x, var_y = np.where(valid_pos, (tx - cx[:, None]) ** 2, 0.0).sum(axis=1) / safe_n_vis, np.where(valid_pos, (ty - cy[:, None]) ** 2, 0.0).sum(axis=1) / safe_n_vis
        feats[f"{prefix}_length_x"], feats[f"{prefix}_width_y"] = (np.sqrt(np.maximum(0.0, var_x)) * 2.0).astype(np.float32), (np.sqrt(np.maximum(0.0, var_y)) * 2.0).astype(np.float32)

        dx_b, dy_b = ball_x[:, None] - tx, ball_y[:, None] - ty
        dist = np.where(valid_pos, np.hypot(dx_b, dy_b), 999.0).astype(np.float32)

        feats[f"{prefix}_in_3m"] = (dist <= CONTROL_RADIUS_3M).sum(axis=1).astype(np.int8)
        feats[f"{prefix}_in_5m"] = (dist <= SUPPORT_RADIUS_5M).sum(axis=1).astype(np.int8)

        proj_x, proj_y = tx + tvx * REACTION_TIME_S, ty + tvy * REACTION_TIME_S
        tti = np.where(valid_pos, REACTION_TIME_S + np.hypot(ball_x[:, None] - proj_x, ball_y[:, None] - proj_y) / PLAYER_MAX_SPEED_MS, 99.0).astype(np.float32)
        feats[f"{prefix}_min_tti_s"] = np.minimum(tti.min(axis=1), 20.0).astype(np.float32)
        pitch_ctrl_rates[prefix] = np.where(valid_pos, np.exp(-tti / PITCH_CONTROL_TAU_S), 0.0).sum(axis=1)

        part_idx = np.argpartition(dist, kth=2, axis=1)[:, :3]
        top3_dist, top3_spd = np.take_along_axis(dist, part_idx, axis=1), np.take_along_axis(ts_z, part_idx, axis=1)
        sub_order = np.argsort(top3_dist, axis=1)
        near1_slot = np.take_along_axis(part_idx, sub_order[:, :1], axis=1)
        sorted_dist, sorted_speed = np.take_along_axis(top3_dist, sub_order, axis=1), np.take_along_axis(top3_spd, sub_order, axis=1)

        for k in range(3):
            feats[f"{prefix}_near{k+1}_dist_m"] = np.minimum(sorted_dist[:, k], 120.0).astype(np.float32)
            feats[f"{prefix}_near{k+1}_speed_ms"] = np.where(sorted_dist[:, k] < 900.0, sorted_speed[:, k], 0.0).astype(np.float32)

        n1_vx, n1_vy = np.take_along_axis(tvx, near1_slot, axis=1)[:, 0], np.take_along_axis(tvy, near1_slot, axis=1)[:, 0]
        n1_dx, n1_dy = np.take_along_axis(np.where(valid_pos, dx_b, 0.0), near1_slot, axis=1)[:, 0], np.take_along_axis(np.where(valid_pos, dy_b, 0.0), near1_slot, axis=1)[:, 0]
        n1_d = feats[f"{prefix}_near1_dist_m"]

        closing_spd = np.where(n1_d < 110.0, ((n1_vx - ball_vx) * n1_dx + (n1_vy - ball_vy) * n1_dy) / (n1_d + 0.1), 0.0)
        feats[f"{prefix}_near1_closing_speed_ms"] = np.clip(closing_spd, -25.0, 25.0).astype(np.float32)
        n1_vnorm = np.hypot(n1_vx, n1_vy)
        vel_cos = np.where((n1_vnorm > 0.3) & (ball_speed > 0.3), (n1_vx * ball_vx + n1_vy * ball_vy) / (n1_vnorm * ball_speed + 1e-5), 0.0)
        feats[f"{prefix}_near1_vel_cos"] = np.clip(vel_cos, -1.0, 1.0).astype(np.float32)

    diff_m = (feats["away_near1_dist_m"] - feats["home_near1_dist_m"]).astype(np.float32)
    feats["near1_dist_diff_m"] = diff_m
    feats["tti_diff_s"] = (feats["away_min_tti_s"] - feats["home_min_tti_s"]).astype(np.float32)

    tot_rate = pitch_ctrl_rates["home"] + pitch_ctrl_rates["away"]
    feats["home_pitch_control"] = np.where(tot_rate > 1e-6, pitch_ctrl_rates["home"] / tot_rate, 0.5).astype(np.float32)

    feats["is_contested"] = ((feats["home_near1_dist_m"] <= CONTESTED_RADIUS_M) & (feats["away_near1_dist_m"] <= CONTESTED_RADIUS_M)).astype(np.int8)

    all_valid_spd = ~np.isnan(ps)
    n_all_spd = np.maximum(1, all_valid_spd.sum(axis=1)).astype(np.float32)
    feats["all_players_mean_speed"] = (np.where(all_valid_spd, ps, 0.0).sum(axis=1) / n_all_spd).astype(np.float32)
    feats["players_walking_ratio"] = ((all_valid_spd & (ps < 1.5)).sum(axis=1) / n_all_spd).astype(np.float32)

    in_left, in_right = (~np.isnan(px)) & (px <= 16.5) & (py >= 13.84) & (py <= 54.16), (~np.isnan(px)) & (px >= PITCH_LENGTH - 16.5) & (py >= 13.84) & (py <= 54.16)
    feats["players_in_penalty_box"] = np.maximum(in_left.sum(axis=1), in_right.sum(axis=1)).astype(np.int8)

    # Use fast causal lfilter for stationary ball history
    is_slow = ((status_code <= 1) & (ball_speed < 1.0)).astype(np.float32)
    feats["ball_stationary_frames"] = lfilter(np.ones(50, dtype=np.float32), 1.0, is_slow).astype(np.int16)

    defl_angle = compute_ball_deflection_angle(ball_x, ball_y, status_code)
    feats["ball_deflection_angle"] = defl_angle
    feats.update(compute_touch_and_control_proxies(feats["home_near1_dist_m"], feats["away_near1_dist_m"], defl_angle, status_code, period))

    # Fully vectorized rolling windows using convolve1d (centered) and lfilter (causal) per half
    roll50, roll100, causal50, causal100 = np.zeros(n_frames, dtype=np.float32), np.zeros(n_frames, dtype=np.float32), np.zeros(n_frames, dtype=np.float32), np.zeros(n_frames, dtype=np.float32)
    raw_obs = np.where(status_code <= 1, diff_m, np.nan)
    
    w50, w100 = np.ones(WINDOW_2S_FRAMES, dtype=np.float32)/WINDOW_2S_FRAMES, np.ones(WINDOW_4S_FRAMES, dtype=np.float32)/WINDOW_4S_FRAMES

    for p_val in (1, 2):
        p_idx = np.flatnonzero(period == p_val)
        if len(p_idx) == 0: continue
        s_half = pd.Series(raw_obs[p_idx]).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
        
        roll50[p_idx] = convolve1d(s_half, w50, axis=0, mode="nearest")
        roll100[p_idx] = convolve1d(s_half, w100, axis=0, mode="nearest")
        causal50[p_idx] = lfilter(w50, 1.0, s_half)
        causal100[p_idx] = lfilter(w100, 1.0, s_half)

    feats.update({"control_diff_roll50": roll50, "control_diff_roll100": roll100, "control_diff_causal50": causal50, "control_diff_causal100": causal100})
    return feats


def verify_and_normalize_halves_and_gk(feats, px, py, slot_teams, ball_x, ball_y, ball_vx, period):
    h1_mask, h2_mask = (period == 1), (period == 2)
    h1_home_x, h1_away_x = float(np.mean(feats["home_centroid_x"][h1_mask])), float(np.mean(feats["away_centroid_x"][h1_mask]))
    h2_home_x, h2_away_x = float(np.mean(feats["home_centroid_x"][h2_mask])), float(np.mean(feats["away_centroid_x"][h2_mask]))

    flip_mask = (h1_mask & (h1_home_x > h1_away_x)) | (h2_mask & (h2_home_x > h2_away_x))
    feats["ball_x_norm"] = np.where(flip_mask, PITCH_LENGTH - ball_x, ball_x).astype(np.float32)
    feats["ball_vx_norm"] = np.where(flip_mask, -ball_vx, ball_vx).astype(np.float32)
    feats["home_centroid_x_norm"] = np.where(flip_mask, PITCH_LENGTH - feats["home_centroid_x"], feats["home_centroid_x"]).astype(np.float32)
    feats["away_centroid_x_norm"] = np.where(flip_mask, PITCH_LENGTH - feats["away_centroid_x"], feats["away_centroid_x"]).astype(np.float32)

    feats["collective_vx_norm"] = (0.5 * (np.where(flip_mask, -feats["home_centroid_vx"], feats["home_centroid_vx"]) + 
                                          np.where(flip_mask, -feats["away_centroid_vx"], feats["away_centroid_vx"]))).astype(np.float32)

    px_norm, bx_norm = np.where(flip_mask[:, None], PITCH_LENGTH - px, px), feats["ball_x_norm"]
    hx_norm, ax_norm = px_norm[:, (slot_teams == HOME)], px_norm[:, (slot_teams == AWAY)]
    hy, ay = py[:, (slot_teams == HOME)], py[:, (slot_teams == AWAY)]

    h_gk_idx = np.argmin(np.where(np.isnan(hx_norm), 999.0, hx_norm), axis=1)[:, None]
    a_gk_idx = np.argmax(np.where(np.isnan(ax_norm), -999.0, ax_norm), axis=1)[:, None]

    h_gk_x, h_gk_y = np.take_along_axis(hx_norm, h_gk_idx, axis=1)[:, 0], np.take_along_axis(hy, h_gk_idx, axis=1)[:, 0]
    a_gk_x, a_gk_y = np.take_along_axis(ax_norm, a_gk_idx, axis=1)[:, 0], np.take_along_axis(ay, a_gk_idx, axis=1)[:, 0]

    feats["home_gk_dist_m"] = np.minimum(np.where(~np.isnan(h_gk_x), np.hypot(bx_norm - h_gk_x, ball_y - h_gk_y), 120.0), 120.0).astype(np.float32)
    feats["away_gk_dist_m"] = np.minimum(np.where(~np.isnan(a_gk_x), np.hypot(bx_norm - a_gk_x, ball_y - a_gk_y), 120.0), 120.0).astype(np.float32)

    h_gk_hold = (feats["home_gk_dist_m"] <= 3.5) & (bx_norm <= 20.0) & (feats["away_near1_dist_m"] > 4.5)
    a_gk_hold = (feats["away_gk_dist_m"] <= 3.5) & (bx_norm >= PITCH_LENGTH - 20.0) & (feats["home_near1_dist_m"] > 4.5)
    feats["gk_holding_ball_proxy"] = np.where(h_gk_hold, 1, np.where(a_gk_hold, -1, 0)).astype(np.int8)


def audit_and_patch_nans(df, max_nan_ratio=0.05):
    n_total = len(df)
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0].index.tolist()

    if not nan_cols:
        print("[CONTRACT AUDIT] Zero NaN values across all columns (100.0% clean).")
        return

    print(f"\n[CONTRACT AUDIT] Detected NaN values in {len(nan_cols)} column(s):")
    for col in nan_cols:
        cnt = int(nan_counts[col])
        pct = 100.0 * cnt / n_total
        print(f"  - {col:<28}: {cnt:>7} / {n_total} frames ({pct:6.3f}%)")
        if (cnt / n_total) > max_nan_ratio:
            raise RuntimeError(f"Column '{col}' has {pct:.2f}% NaN values, exceeding safety threshold ({100.0 * max_nan_ratio:.1f}%)!")

    df[nan_cols] = df[nan_cols].ffill().bfill().fillna(0.0)
    assert not df.columns[df.isna().any()].tolist(), "NaN values still present after patching!"


# ---------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------
def build_stage1():
    t0 = time.time()
    meta_path, raw_path, st_path, homo_path, n_frames, (h1_s, h1_e, h2_s, h2_e) = resolve_match_paths()
    slot_teams = get_slot_teams(meta_path)
    
    print(f"\n[1/5] Loading {n_frames} frames ({len(slot_teams)} slots) from {os.path.basename(raw_path)}...")
    px, py, ps, bx, by = parse_raw_atd(raw_path, n_frames, len(slot_teams))
    px, py, ps = heal_player_tracks(px, py, ps, slot_teams)
    px, py, ps, pvx, pvy = compute_player_velocities_savgol(px, py, ps)

    print("[2/5] Computing Focal Action Centroid, gap timers, and Savitzky-Golay ball kinematics...")
    att_x, att_y, att_spread = compute_focal_attention_centroid(px, py, ps, pvx, pvy)
    ball_x, ball_y, status_code, status_label, sigma_m, gap_len, f_since, f_to, out = impute_ball_and_uncertainty(bx, by, att_x, att_y, att_spread)
    ball_vx, ball_vy, ball_speed, ball_acc = compute_clean_ball_kinematics(ball_x, ball_y, status_code)

    frames, period = np.arange(n_frames, dtype=np.int32), np.zeros(n_frames, dtype=np.int8)
    period[(frames >= h1_s) & (frames <= h1_e)], period[(frames >= h2_s) & (frames <= h2_e)] = 1, 2
    is_graded, is_pre_kickoff_d = (period > 0).astype(np.int8), ((frames >= h2_s) & (frames < KICKOFF_2ND_HALF)).astype(np.int8)

    print("[3/5] Extracting canonical team features, GK proxy, Dead-Ball heuristics, and dual rolling windows...")
    team_feats = extract_canonical_team_features(px, py, ps, pvx, pvy, slot_teams, ball_x, ball_y, ball_vx, ball_vy, ball_speed, status_code, period)
    verify_and_normalize_halves_and_gk(team_feats, px, py, slot_teams, ball_x, ball_y, ball_vx, period)

    print("[4/5] Integrating Smart Tagging priors & Homography camera dynamics...")
    st_feats = resolve_smart_tagging_na_and_enhance(
        st_path, n_frames, 
        team_feats["home_near1_dist_m"], team_feats["away_near1_dist_m"], 
        team_feats["home_centroid_y"], team_feats["away_centroid_y"],
        ball_speed, status_code
    )
    homo_feats = extract_homography_features(homo_path, n_frames, ball_x, ball_y)

    df = pd.DataFrame({
        "frame_id": frames, "period": period, "is_graded": is_graded, "is_pre_kickoff_d": is_pre_kickoff_d,
        "timestamp_s": (frames * DT).astype(np.float32),
        "ball_x": ball_x, "ball_y": ball_y, "ball_vx": ball_vx, "ball_vy": ball_vy,
        "ball_speed": ball_speed, "ball_acc": ball_acc,
        "ball_status": status_label, "ball_status_code": status_code,
        "ball_sigma_m": sigma_m, "ball_gap_frames": gap_len,
        "frames_since_ball_gap": f_since, "frames_to_ball_gap": f_to, "ball_near_boundary": out,
        **homo_feats, **st_feats, **team_feats,
    })

    audit_and_patch_nans(df)
    out_path = os.path.join(ROOT, "data", "interim", "stage1_tracking.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n[5/5] SUCCESS! Saved {out_path} ({len(df.columns)} columns) in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    build_stage1()