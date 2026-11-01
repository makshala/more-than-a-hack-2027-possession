"""
Role 2 (Tracking): Generating the data contract data/interim/stage1_tracking.parquet (v3.0 Top-2% Pure Tactical).

Key Architecture & Top-2% Engineering Features:
  1. Strict separation of concerns: 100% deterministic tactical timeline processing (Broadcast fusion
     decoupled to a standalone R&D module).
  2. Segment-isolated ball kinematics: computes ball_vx, ball_vy, ball_speed, and ball_acc without
     differentiating across OBSERVED <-> INFERRED_LATENT boundaries (eliminates boundary velocity spikes).
  3. Smart Tagging DEFEND* polarity inversion + Sideline-aware Throw-in N/A resolver:
     correctly assigns defensive tags to the attacking opponent and resolves all 36 'Team = N/A' restarts.
  4. Vector player kinematics (pvx, pvy), radial closing speed, velocity cosine similarity,
     and boundary-safe ball deflection angle.
  5. Domain-grounded Pitch Control (TTI with 0.4s reaction time), Focal Action Centroid, and
     CHALLENGE.md controlled possession proxies (consecutive touches & last controlling team).
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

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

# Domain-grounded physical constants (independent of public GT)
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

# Official frame boundaries from DATA.md and FAQ.md
H1_START, H1_END = 11185, 90651
H2_START, H2_END = 110808, 190197
KICKOFF_2ND_HALF = 113742

HOME = 0
AWAY = 1


def resolve_match_paths():
    """Resolves tactical asset paths via games.py with standard data/ fallback."""
    base_dir = os.environ.get("MTAH_DATA_DIR", os.path.join(ROOT, "data"))
    tac_dir = os.path.join(base_dir, "tactical")

    if games is not None and hasattr(games, "asset_paths"):
        t_paths = games.asset_paths(game=GAME)
        meta_path = t_paths["atd_meta"]
        raw_path = t_paths["atd_raw"]
        st_path = t_paths.get("smart_tagging", os.path.join(tac_dir, f"{GAME}_tactical_smart_tagging.xml"))
    else:
        meta_path = os.path.join(tac_dir, f"{GAME}_tactical_atd.xml")
        raw_path = os.path.join(tac_dir, f"{GAME}_tactical_atd.txt")
        st_path = os.path.join(tac_dir, f"{GAME}_tactical_smart_tagging.xml")

    if games is not None and hasattr(games, "half_video_blocks"):
        blocks = games.half_video_blocks(GAME)
        (_, h1_s, h1_e), (_, h2_s, h2_e) = blocks[0], blocks[1]
    else:
        h1_s, h1_e, h2_s, h2_e = H1_START, H1_END, H2_START, H2_END

    if games is not None and hasattr(games, "source_frames"):
        n_frames = int(games.source_frames("tactical", GAME))
    else:
        n_frames = 199319

    return meta_path, raw_path, st_path, n_frames, (h1_s, h1_e, h2_s, h2_e)


def get_slot_teams(meta_path):
    """Maps each player slot by its ShirtNumber (1-based slot index) and verifies HOME == Barcelona."""
    root = ET.parse(meta_path).getroot()
    team_info = {}
    for team in root.iter("Team"):
        tid = team.get("id") or team.get("Id") or ""
        tname = team.get("name") or team.get("Name") or (team.text or "").strip()
        for child in team:
            if "name" in child.tag.lower() and child.text:
                tname = child.text.strip()
        team_info[tid] = tname

    home_id = None
    for tid, tname in team_info.items():
        label_str = f"{tid} {tname}".lower()
        if "bar" in label_str or "fcb" in label_str or "home" in label_str:
            home_id = tid
            break
    if home_id is None:
        home_id = list(team_info.keys())[0]

    players = list(root.iter("Player"))
    n_slots = len(players)
    slot_teams = np.full(n_slots, -1, dtype=np.int8)

    for idx, player in enumerate(players):
        tid = player.get("teamId") or player.get("TeamId")
        shirt_str = player.get("shirtNumber") or player.get("ShirtNumber")
        if shirt_str is None:
            for child in player:
                if "shirtnumber" in child.tag.lower() and child.text:
                    shirt_str = child.text.strip()
        try:
            slot_idx = int(shirt_str) - 1 if shirt_str is not None else idx
            if not (0 <= slot_idx < n_slots):
                slot_idx = idx
        except ValueError:
            slot_idx = idx
        slot_teams[slot_idx] = HOME if tid == home_id else AWAY

    if np.any(slot_teams < 0):
        slot_teams = np.array(
            [HOME if (p.get("teamId") or p.get("TeamId")) == home_id else AWAY for p in players],
            dtype=np.int8,
        )
    return slot_teams


def parse_raw_atd(raw_path, n_frames, n_slots):
    """Streams raw ATD text file and scales normalized coordinates to meters."""
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)

    plen, pwid = PITCH_LENGTH, PITCH_WIDTH

    with open(raw_path, "r") as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frame_idx = int(parts[0]) - 1
            if not (0 <= frame_idx < n_frames):
                continue

            for slot, group in enumerate(parts[1].split(";")):
                if slot >= n_slots or not group or group[0] == "N":
                    continue
                c1 = group.find(",")
                c2 = group.find(",", c1 + 1)
                px[frame_idx, slot] = float(group[:c1]) * plen
                py[frame_idx, slot] = float(group[c1 + 1:c2]) * pwid
                s_str = group[c2 + 1:]
                if s_str and s_str[0] != "N":
                    ps[frame_idx, slot] = float(s_str)

            ball = parts[2]
            if ball and ball[0] != "N":
                c1 = ball.find(",")
                c2 = ball.find(",", c1 + 1)
                bx[frame_idx] = float(ball[:c1]) * plen
                by[frame_idx] = float(ball[c1 + 1:c2]) * pwid

    # Repair missing player speeds where position (px, py) is valid
    pos_valid = ~np.isnan(px)
    speed_missing = pos_valid & np.isnan(ps)
    if speed_missing.any():
        dx = np.diff(px, axis=0, prepend=px[:1])
        dy = np.diff(py, axis=0, prepend=py[:1])
        kin_speed = np.hypot(dx, dy) / DT
        ps[speed_missing] = np.where(
            np.isnan(kin_speed[speed_missing]), 0.0, np.minimum(kin_speed[speed_missing], 12.0)
        )

    return px, py, ps, bx, by


# ---------------------------------------------------------
# 2. TRACKLET HEALING, DEDUPLICATION & PLAYER VELOCITIES
# ---------------------------------------------------------
def heal_player_tracks(px, py, ps, slot_teams, max_stitch_frames=25):
    """
    Heals identity handoffs and suppresses duplicate clone slots:
      1. Inter-slot Hungarian stitching (linear_sum_assignment) on contiguous tracklet segments.
      2. Spatial clone deduplication (< 0.45m within same team).
    """
    n_frames, _ = px.shape
    px_h, py_h, ps_h = px.copy(), py.copy(), ps.copy()
    total_stitched = 0
    total_clones_removed = 0

    for team_code in (HOME, AWAY):
        team_slots = np.flatnonzero(slot_teams == team_code)
        deaths = []
        births = []

        for s in team_slots:
            valid = ~np.isnan(px_h[:, s])
            if valid.sum() < 15:
                continue
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
            cost_mat = np.full((n_d, n_b), 1e6, dtype=np.float32)

            for i, (d_slot, d_f, dx, dy, dvx, dvy) in enumerate(deaths):
                for j, (b_slot, b_f, b_end, bx_val, by_val) in enumerate(births):
                    gap = b_f - d_f
                    if b_slot != d_slot and 0 < gap <= max_stitch_frames:
                        dt_gap = gap * DT
                        dist = float(np.hypot(bx_val - (dx + dvx * dt_gap), by_val - (dy + dvy * dt_gap)))
                        if dist <= 2.5:
                            cost_mat[i, j] = dist

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            used_slots = set()

            for r, c in zip(row_ind, col_ind):
                if cost_mat[r, c] >= 2.5:
                    continue
                d_slot, d_f, dx, dy, dvx, dvy = deaths[r]
                b_slot, b_f, b_end, bx_val, by_val = births[c]
                if d_slot in used_slots or b_slot in used_slots:
                    continue

                if not np.all(np.isnan(px_h[d_f + 1:b_end + 1, d_slot])):
                    continue

                gap_len = b_f - d_f
                alpha = np.linspace(0.0, 1.0, gap_len + 1, dtype=np.float32)[1:-1]
                if len(alpha) > 0:
                    px_h[d_f + 1:b_f, d_slot] = (1.0 - alpha) * dx + alpha * bx_val
                    py_h[d_f + 1:b_f, d_slot] = (1.0 - alpha) * dy + alpha * by_val
                    ps_h[d_f + 1:b_f, d_slot] = (1.0 - alpha) * np.hypot(dvx, dvy) + alpha * ps_h[b_f, b_slot]

                seg_slice = slice(b_f, b_end + 1)
                px_h[seg_slice, d_slot] = px_h[seg_slice, b_slot]
                py_h[seg_slice, d_slot] = py_h[seg_slice, b_slot]
                ps_h[seg_slice, d_slot] = ps_h[seg_slice, b_slot]

                px_h[seg_slice, b_slot] = np.nan
                py_h[seg_slice, b_slot] = np.nan
                ps_h[seg_slice, b_slot] = np.nan

                used_slots.add(d_slot)
                used_slots.add(b_slot)
                total_stitched += 1

        # Spatial clone deduplication (< 0.45m within same team)
        tx = px_h[:, team_slots]
        ty = py_h[:, team_slots]
        m_team = len(team_slots)
        for i in range(m_team):
            for j in range(i + 1, m_team):
                dist_ij = np.hypot(tx[:, i] - tx[:, j], ty[:, i] - ty[:, j])
                clones = dist_ij < CLONE_DIST_THRESHOLD_M
                if clones.any():
                    s_drop = team_slots[j]
                    px_h[clones, s_drop] = np.nan
                    py_h[clones, s_drop] = np.nan
                    ps_h[clones, s_drop] = np.nan
                    tx[clones, j] = np.nan
                    ty[clones, j] = np.nan
                    total_clones_removed += int(clones.sum())

    print(f"[TRACKLET HEALER] Hungarian stitched {total_stitched} tracklets; removed {total_clones_removed} duplicate clone observations.")
    return px_h, py_h, ps_h


def compute_player_velocities(px, py):
    """Computes clean per-slot 2D velocity vectors (pvx, pvy) in m/s without NaN boundary spikes."""
    dx_f = np.diff(px, axis=0, append=np.nan)
    dy_f = np.diff(py, axis=0, append=np.nan)
    dx_b = np.diff(px, axis=0, prepend=np.nan)
    dy_b = np.diff(py, axis=0, prepend=np.nan)

    has_f = ~np.isnan(dx_f)
    has_b = ~np.isnan(dx_b)

    vx = np.where(
        has_f & has_b, 0.5 * (dx_f + dx_b) / DT,
        np.where(has_f, dx_f / DT, np.where(has_b, dx_b / DT, 0.0))
    ).astype(np.float32)
    vy = np.where(
        has_f & has_b, 0.5 * (dy_f + dy_b) / DT,
        np.where(has_f, dy_f / DT, np.where(has_b, dy_b / DT, 0.0))
    ).astype(np.float32)

    unphys = (np.hypot(vx, vy) > 12.5) | np.isnan(px)
    vx[unphys] = 0.0
    vy[unphys] = 0.0
    return vx, vy


# ---------------------------------------------------------
# 3. FOCAL BALL IMPUTATION & BOUNDARY-SAFE BALL KINEMATICS
# ---------------------------------------------------------
def compute_focal_attention_centroid(px, py, ps, pvx, pvy):
    """
    Estimates the Virtual Ball (Focal Action Centroid) by combining:
      1. Logarithmic speed weight: 1 + log1p(speed)
      2. Spatial cluster kernel (18m) + convergence speed toward the active cluster,
         preventing isolated goalkeepers and far-side defenders from pulling the ball to the center.
    """
    valid = ~np.isnan(px)
    n_visible = valid.sum(axis=1)
    px_z = np.where(valid, px, 0.0)
    py_z = np.where(valid, py, 0.0)

    spd = np.maximum(0.0, np.nan_to_num(ps, nan=0.0))
    w_coarse = np.where(valid, 1.0 + np.log1p(spd), 0.0).astype(np.float32)
    w_sum1 = np.maximum(1e-5, w_coarse.sum(axis=1))
    cx1 = np.where(n_visible > 0, (px_z * w_coarse).sum(axis=1) / w_sum1, PITCH_LENGTH / 2.0)
    cy1 = np.where(n_visible > 0, (py_z * w_coarse).sum(axis=1) / w_sum1, PITCH_WIDTH / 2.0)

    dx1 = np.where(valid, cx1[:, None] - px_z, 0.0)
    dy1 = np.where(valid, cy1[:, None] - py_z, 0.0)
    dist1 = np.hypot(dx1, dy1)

    conv_speed = np.maximum(0.0, (pvx * dx1 + pvy * dy1) / (dist1 + 0.5))
    spatial_kernel = np.exp(-0.5 * (dist1 / 18.0) ** 2)
    w_focal = np.where(valid, w_coarse * (0.35 + 0.65 * spatial_kernel) * (1.0 + 0.25 * conv_speed), 0.0).astype(np.float32)

    w_sum2 = np.maximum(1e-5, w_focal.sum(axis=1))
    att_x = np.where(n_visible > 0, (px_z * w_focal).sum(axis=1) / w_sum2, PITCH_LENGTH / 2.0).astype(np.float32)
    att_y = np.where(n_visible > 0, (py_z * w_focal).sum(axis=1) / w_sum2, PITCH_WIDTH / 2.0).astype(np.float32)

    dx2 = np.where(valid, px_z - att_x[:, None], 0.0)
    dy2 = np.where(valid, py_z - att_y[:, None], 0.0)
    spread_m = np.sqrt(np.where(n_visible > 0, ((dx2 ** 2 + dy2 ** 2) * w_focal).sum(axis=1) / w_sum2, 100.0)).astype(np.float32)

    return att_x, att_y, spread_m


def compute_gap_boundary_timers(status_code):
    """Vectorized calculation of frame distance to the nearest long gap (status_code == 2)."""
    n_frames = len(status_code)
    is_long_gap = (status_code == 2)
    gap_indices = np.flatnonzero(is_long_gap)

    frames_since = np.full(n_frames, 10000, dtype=np.int32)
    frames_to = np.full(n_frames, 10000, dtype=np.int32)

    if len(gap_indices) == 0:
        return frames_since, frames_to

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
    """
    Classifies frames into OBSERVED (0), INTERPOLATED_SHORT (1), INFERRED_LATENT (2).
    In INFERRED_LATENT gaps, smoothly blends the endpoint linear bridge with the Focal Action Centroid.
    """
    n_frames = len(bx)
    observed = ~np.isnan(bx)

    ball_x, ball_y = bx.copy(), by.copy()
    status_code = np.full(n_frames, 2, dtype=np.int8)
    status_code[observed] = 0
    sigma_m = np.full(n_frames, 8.0, dtype=np.float32)
    sigma_m[observed] = 0.5
    gap_length_arr = np.zeros(n_frames, dtype=np.int32)

    padded = np.concatenate(([True], observed, [True]))
    gap_starts = np.flatnonzero(padded[:-1] & ~padded[1:])
    gap_ends = np.flatnonzero(~padded[:-1] & padded[1:])

    for start, end in zip(gap_starts, gap_ends):
        gap_len = end - start
        gap_length_arr[start:end] = gap_len
        has_left, has_right = start > 0, end < n_frames

        if gap_len <= SHORT_GAP_MAX_FRAMES and has_left and has_right:
            status_code[start:end] = 1
            alpha = np.linspace(0.0, 1.0, gap_len + 2, dtype=np.float32)[1:-1]
            ball_x[start:end] = (1.0 - alpha) * bx[start - 1] + alpha * bx[end]
            ball_y[start:end] = (1.0 - alpha) * by[start - 1] + alpha * by[end]
            dist_to_edge = np.minimum(np.arange(1, gap_len + 1), np.arange(gap_len, 0, -1))
            sigma_m[start:end] = 0.5 + 0.16 * dist_to_edge
        else:
            status_code[start:end] = 2
            est_x, est_y = att_x[start:end].copy(), att_y[start:end].copy()
            dist_to_edge = np.minimum(np.arange(1, gap_len + 1), np.arange(gap_len, 0, -1)).astype(np.float32)

            if has_left and has_right:
                alpha = np.linspace(0.0, 1.0, gap_len + 2, dtype=np.float32)[1:-1]
                bridge_x = (1.0 - alpha) * bx[start - 1] + alpha * bx[end]
                bridge_y = (1.0 - alpha) * by[start - 1] + alpha * by[end]
                w_bridge = np.exp(-dist_to_edge / 40.0).astype(np.float32)
                est_x = w_bridge * bridge_x + (1.0 - w_bridge) * est_x
                est_y = w_bridge * bridge_y + (1.0 - w_bridge) * est_y
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

            ball_x[start:end] = est_x
            ball_y[start:end] = est_y
            sigma_m[start:end] = np.clip(0.32 * att_spread[start:end] * (1.0 - 0.5 * np.exp(-dist_to_edge / 35.0)), 2.5, 25.0)

    status_label = np.array(["OBSERVED", "INTERPOLATED_SHORT", "INFERRED_LATENT"])[status_code]
    frames_since_gap, frames_to_gap = compute_gap_boundary_timers(status_code)
    out_of_bounds = (
        (ball_x < 0.5) | (ball_x > PITCH_LENGTH - 0.5) |
        (ball_y < 0.5) | (ball_y > PITCH_WIDTH - 0.5)
    ).astype(np.int8)

    return ball_x, ball_y, status_code, status_label, sigma_m, gap_length_arr, frames_since_gap, frames_to_gap, out_of_bounds


def compute_clean_ball_kinematics(ball_x, ball_y, status_code):
    """
    Computes ball velocity (vx, vy, speed) and acceleration (acc) WITHOUT differentiating
    across the seam between OBSERVED/SHORT (<=1) and INFERRED_LATENT (2) segments.
    Prevents artificial 100+ m/s velocity spikes at occlusion boundaries.
    """
    is_obs = (status_code <= 1)
    same_next = (is_obs[:-1] == is_obs[1:])
    has_f = np.concatenate((same_next, [False]))
    has_b = np.concatenate(([False], same_next))

    dx_f = np.diff(ball_x, append=ball_x[-1])
    dy_f = np.diff(ball_y, append=ball_y[-1])
    dx_b = np.diff(ball_x, prepend=ball_x[0])
    dy_b = np.diff(ball_y, prepend=ball_y[0])

    vx = np.where(
        has_f & has_b, 0.5 * (dx_f + dx_b) / DT,
        np.where(has_f, dx_f / DT, np.where(has_b, dx_b / DT, 0.0))
    ).astype(np.float32)
    vy = np.where(
        has_f & has_b, 0.5 * (dy_f + dy_b) / DT,
        np.where(has_f, dy_f / DT, np.where(has_b, dy_b / DT, 0.0))
    ).astype(np.float32)

    speed = np.hypot(vx, vy).astype(np.float32)
    # Clamp unphysical detector jumps (> 38 m/s)
    over_max = speed > MAX_BALL_SPEED_MS
    if over_max.any():
        scale = MAX_BALL_SPEED_MS / np.maximum(speed, 1e-5)
        vx = np.where(over_max, vx * scale, vx).astype(np.float32)
        vy = np.where(over_max, vy * scale, vy).astype(np.float32)
        speed = np.minimum(speed, MAX_BALL_SPEED_MS).astype(np.float32)

    dvx_f = np.diff(vx, append=vx[-1])
    dvy_f = np.diff(vy, append=vy[-1])
    dvx_b = np.diff(vx, prepend=vx[0])
    dvy_b = np.diff(vy, prepend=vy[0])

    ax = np.where(has_f & has_b, 0.5 * (dvx_f + dvx_b) / DT, np.where(has_f, dvx_f / DT, 0.0))
    ay = np.where(has_f & has_b, 0.5 * (dvy_f + dvy_b) / DT, np.where(has_f, dvy_f / DT, 0.0))
    acc = np.minimum(np.hypot(ax, ay), 150.0).astype(np.float32)

    return vx, vy, speed, acc


# ---------------------------------------------------------
# 4. SMART TAGGING POLARITY FIX & SIDELINE RESTART RESOLVER
# ---------------------------------------------------------
def resolve_smart_tagging_na_and_enhance(st_path, n_frames, px, py, slot_teams, ball_x, ball_y):
    """
    Parses Smart Tagging XML with three Top-2% enhancements:
      1. Fixes DEFEND* tag polarity: when code starts with 'DEFEND', the tagged team is defending,
         so the opposing team is in possession (+1 vote to opponent).
      2. Resolves all 36 'Team = N/A' instances (including all 26 Throw-ins) using sideline-aware
         player-ball restart geometry, and outputs explicit st_restart_team (+1 HOME, -1 AWAY).
      3. Exposes st_defending_tag, st_in_box_tag, and continuous st_team_score.
    """
    st_set_piece = np.zeros(n_frames, dtype=np.int8)
    st_throw_in = np.zeros(n_frames, dtype=np.int8)
    st_defending_tag = np.zeros(n_frames, dtype=np.int8)
    st_in_box_tag = np.zeros(n_frames, dtype=np.int8)
    st_restart_team = np.zeros(n_frames, dtype=np.int8)
    home_votes = np.zeros(n_frames, dtype=np.float32)
    away_votes = np.zeros(n_frames, dtype=np.float32)
    st_n_active = np.zeros(n_frames, dtype=np.int8)

    if not os.path.exists(st_path):
        return {
            "st_set_piece": st_set_piece,
            "st_throw_in": st_throw_in,
            "st_defending_tag": st_defending_tag,
            "st_in_box_tag": st_in_box_tag,
            "st_restart_team": st_restart_team,
            "st_team_prior": np.zeros(n_frames, dtype=np.int8),
            "st_team_score": np.zeros(n_frames, dtype=np.float32),
            "st_n_active_tags": st_n_active,
        }

    home_mask = (slot_teams == HOME)
    away_mask = (slot_teams == AWAY)
    resolved_na_count = 0
    inverted_def_count = 0

    root = ET.parse(st_path).getroot()
    for inst in root.iter("instance"):
        start_el, end_el, code_el = inst.find("start"), inst.find("end"), inst.find("code")
        if start_el is None or end_el is None or code_el is None:
            continue

        s_frame = max(0, min(n_frames - 1, int(round(float(start_el.text) * FPS))))
        e_frame = max(0, min(n_frames, int(round(float(end_el.text) * FPS)) + 1))
        code = (code_el.text or "").strip().upper()

        tags = {
            label.find("group").text.strip().lower(): label.find("text").text.strip()
            for label in inst.findall("label")
            if label.find("group") is not None and label.find("text") is not None
            and label.find("group").text and label.find("text").text
        }

        team_val = tags.get("team", "N/A").lower()
        type_val = tags.get("type", "").lower()
        st_n_active[s_frame:e_frame] = np.minimum(120, st_n_active[s_frame:e_frame] + 1)

        is_defending_code = code.startswith("DEFEND")
        if is_defending_code:
            st_defending_tag[s_frame:e_frame] = 1
        if "IN THE BOX" in code:
            st_in_box_tag[s_frame:e_frame] = 1

        is_set_piece = (code == "SET PIECES")
        is_throw_in = ("throw-in" in type_val)
        if is_set_piece:
            st_set_piece[s_frame:e_frame] = 1
            if is_throw_in:
                st_throw_in[s_frame:e_frame] = 1

        restart_window_end = min(n_frames, e_frame + 35)

        if "barcelona" in team_val:
            if is_defending_code:
                away_votes[s_frame:e_frame] += 1.0
                inverted_def_count += 1
            else:
                home_votes[s_frame:e_frame] += 1.0
                if is_set_piece:
                    st_restart_team[max(0, e_frame - 10):restart_window_end] = 1
        elif "atletic" in team_val or "athletic" in team_val:
            if is_defending_code:
                home_votes[s_frame:e_frame] += 1.0
                inverted_def_count += 1
            else:
                away_votes[s_frame:e_frame] += 1.0
                if is_set_piece:
                    st_restart_team[max(0, e_frame - 10):restart_window_end] = -1
        else:
            w_start, w_end = max(0, e_frame - 20), min(n_frames, e_frame + 25)
            if w_end > w_start:
                bx_w, by_w = ball_x[w_start:w_end, None], ball_y[w_start:w_end, None]
                hx_w, hy_w = px[w_start:w_end][:, home_mask], py[w_start:w_end][:, home_mask]
                ax_w, ay_w = px[w_start:w_end][:, away_mask], py[w_start:w_end][:, away_mask]

                h_d = np.hypot(hx_w - bx_w, hy_w - by_w)
                a_d = np.hypot(ax_w - bx_w, ay_w - by_w)

                # For Throw-ins, weight players standing right on the touchline (y < 2.5m or y > 65.5m)
                if is_throw_in:
                    h_sideline_dist = np.minimum(hy_w, PITCH_WIDTH - hy_w)
                    a_sideline_dist = np.minimum(ay_w, PITCH_WIDTH - ay_w)
                    h_d = h_d + 0.5 * np.maximum(0.0, h_sideline_dist - 1.5)
                    a_d = a_d + 0.5 * np.maximum(0.0, a_sideline_dist - 1.5)

                h_med = float(np.nanmedian(np.where(np.all(np.isnan(h_d), axis=1), 999.0, np.nanmin(np.where(np.isnan(h_d), 999.0, h_d), axis=1))))
                a_med = float(np.nanmedian(np.where(np.all(np.isnan(a_d), axis=1), 999.0, np.nanmin(np.where(np.isnan(a_d), 999.0, a_d), axis=1))))

                if h_med <= a_med:
                    home_votes[s_frame:restart_window_end] += 0.85
                    if is_set_piece:
                        st_restart_team[max(0, e_frame - 10):restart_window_end] = 1
                else:
                    away_votes[s_frame:restart_window_end] += 0.85
                    if is_set_piece:
                        st_restart_team[max(0, e_frame - 10):restart_window_end] = -1
                resolved_na_count += 1

    net_score = (home_votes - away_votes).astype(np.float32)
    print(f"[SMART TAGGING ENHANCER] Inverted polarity on {inverted_def_count} DEFEND* tags; resolved {resolved_na_count} 'Team=N/A' instances.")
    return {
        "st_set_piece": st_set_piece,
        "st_throw_in": st_throw_in,
        "st_defending_tag": st_defending_tag,
        "st_in_box_tag": st_in_box_tag,
        "st_restart_team": st_restart_team,
        "st_team_prior": np.sign(net_score).astype(np.int8),
        "st_team_score": net_score,
        "st_n_active_tags": st_n_active,
    }


# ---------------------------------------------------------
# 5. KINEMATICS, TTI, PITCH CONTROL & TOUCH PROXIES
# ---------------------------------------------------------
def compute_ball_deflection_angle(ball_x, ball_y, status_code):
    """
    Computes instantaneous directional change (in radians, 0..pi) of the ball trajectory
    over a +/- 2 frame window, strictly requiring all 3 points (t-2, t, t+2) to be observed/short.
    """
    obs = (status_code <= 1)
    obs_prev2 = np.roll(obs, 2)
    obs_next2 = np.roll(obs, -2)
    obs_prev2[:2] = False
    obs_next2[-2:] = False

    dx_in = ball_x - np.roll(ball_x, 2)
    dy_in = ball_y - np.roll(ball_y, 2)
    dx_out = np.roll(ball_x, -2) - ball_x
    dy_out = np.roll(ball_y, -2) - ball_y

    norm_in = np.hypot(dx_in, dy_in)
    norm_out = np.hypot(dx_out, dy_out)

    valid_defl = obs & obs_prev2 & obs_next2 & (norm_in > 0.25) & (norm_out > 0.25)
    safe_denom = np.where(valid_defl, norm_in * norm_out, 1.0)
    cos_sim = np.clip((dx_in * dx_out + dy_in * dy_out) / safe_denom, -1.0, 1.0)
    return np.where(valid_defl, np.arccos(cos_sim), 0.0).astype(np.float32)


def compute_touch_and_control_proxies(home_d1, away_d1, defl_angle, status_code, period):
    """
    Implements CHALLENGE.md possession rules strictly per half:
      - Distinguishes brief contested rebounds/deflections from controlled touches.
      - Tracks consecutive controlled touch episodes and last controlling team (+1 HOME, -1 AWAY).
    """
    n_frames = len(home_d1)
    obs = (status_code <= 1)

    # Uncontested close proximity (< 1.8m while opponent > 2.0m)
    h_uncontested = obs & (home_d1 <= 1.8) & (away_d1 > 2.0)
    a_uncontested = obs & (away_d1 <= 1.8) & (home_d1 > 2.0)

    home_consec_touches = np.zeros(n_frames, dtype=np.int16)
    away_consec_touches = np.zeros(n_frames, dtype=np.int16)
    last_controlled_team = np.zeros(n_frames, dtype=np.int8)
    frames_since_control = np.full(n_frames, 10000, dtype=np.int32)

    for p_val in (1, 2):
        p_idx = np.flatnonzero(period == p_val)
        if len(p_idx) == 0:
            continue
        s, e = int(p_idx[0]), int(p_idx[-1]) + 1
        hc, ac = h_uncontested[s:e], a_uncontested[s:e]
        defl_sub = defl_angle[s:e]
        m = e - s

        # Sustained proximity (>= 4 frames within an 8-frame window) overrides any deflection angle
        hc_sust = pd.Series(hc.astype(np.int8)).rolling(8, min_periods=1).sum().to_numpy() >= 4
        ac_sust = pd.Series(ac.astype(np.int8)).rolling(8, min_periods=1).sum().to_numpy() >= 4

        # Filter out isolated 1-frame sharp deflections (< 4 frames and angle > 1.3 rad)
        hc_valid = hc_sust | (hc & (defl_sub < 1.3))
        ac_valid = ac_sust | (ac & (defl_sub < 1.3))

        h_cnt_arr = np.zeros(m, dtype=np.int16)
        a_cnt_arr = np.zeros(m, dtype=np.int16)
        team_arr = np.zeros(m, dtype=np.int8)
        since_arr = np.full(m, 10000, dtype=np.int32)

        cur_team, h_cnt, a_cnt, since_ctrl = 0, 0, 0, 10000
        active_frames = np.flatnonzero(hc_valid | ac_valid)
        last_f = 0

        for f in active_frames:
            if f > last_f:
                h_cnt_arr[last_f:f] = h_cnt
                a_cnt_arr[last_f:f] = a_cnt
                team_arr[last_f:f] = cur_team
                if since_ctrl < 10000:
                    span = f - last_f
                    since_arr[last_f:f] = np.minimum(10000, since_ctrl + np.arange(span, dtype=np.int32))
                    since_ctrl = min(10000, since_ctrl + span)

            is_h_ep = hc_valid[f] and (f == 0 or not hc_valid[f - 1])
            is_a_ep = ac_valid[f] and (f == 0 or not ac_valid[f - 1])

            if is_h_ep:
                h_cnt = min(25, h_cnt + 1)
                if hc_sust[f] or h_cnt >= 2:
                    a_cnt = 0
                    cur_team = 1
                    since_ctrl = 0
            elif hc_sust[f]:
                h_cnt = max(1, h_cnt)
                a_cnt = 0
                cur_team = 1
                since_ctrl = 0

            if is_a_ep:
                a_cnt = min(25, a_cnt + 1)
                if ac_sust[f] or a_cnt >= 2:
                    h_cnt = 0
                    cur_team = -1
                    since_ctrl = 0
            elif ac_sust[f]:
                a_cnt = max(1, a_cnt)
                h_cnt = 0
                cur_team = -1
                since_ctrl = 0

            h_cnt_arr[f] = h_cnt
            a_cnt_arr[f] = a_cnt
            team_arr[f] = cur_team
            since_arr[f] = since_ctrl
            since_ctrl = min(10000, since_ctrl + 1)
            last_f = f + 1

        if last_f < m:
            h_cnt_arr[last_f:m] = h_cnt
            a_cnt_arr[last_f:m] = a_cnt
            team_arr[last_f:m] = cur_team
            if since_ctrl < 10000:
                since_arr[last_f:m] = np.minimum(10000, since_ctrl + np.arange(m - last_f, dtype=np.int32))

        home_consec_touches[s:e] = h_cnt_arr
        away_consec_touches[s:e] = a_cnt_arr
        last_controlled_team[s:e] = team_arr
        frames_since_control[s:e] = since_arr

    return {
        "home_consecutive_touches": home_consec_touches,
        "away_consecutive_touches": away_consec_touches,
        "last_controlled_team": last_controlled_team,
        "frames_since_last_control": frames_since_control,
    }


def extract_canonical_team_features(
    px, py, ps, pvx, pvy, slot_teams, ball_x, ball_y, ball_vx, ball_vy, ball_speed, status_code, period
):
    """
    Extracts canonical team features, closing speeds, velocity cosine alignment,
    Time-to-Intercept (TTI), continuous Pitch Control, and half-isolated rolling windows.
    """
    feats = {}
    n_frames = px.shape[0]
    pitch_ctrl_rates = {}

    for team_code, prefix in ((HOME, "home"), (AWAY, "away")):
        mask = (slot_teams == team_code)
        tx, ty, ts = px[:, mask], py[:, mask], ps[:, mask]
        tvx, tvy = pvx[:, mask], pvy[:, mask]

        valid_pos = ~np.isnan(tx)
        valid_spd = ~np.isnan(ts)
        n_vis = valid_pos.sum(axis=1).astype(np.int8)
        n_spd = valid_spd.sum(axis=1)
        feats[f"{prefix}_n_visible"] = n_vis

        safe_n_vis = np.maximum(1, n_vis).astype(np.float32)
        safe_n_spd = np.maximum(1, n_spd).astype(np.float32)

        tx_zero = np.where(valid_pos, tx, 0.0)
        ty_zero = np.where(valid_pos, ty, 0.0)
        ts_zero = np.where(valid_spd, ts, 0.0)

        cx = np.where(n_vis > 0, tx_zero.sum(axis=1) / safe_n_vis, PITCH_LENGTH / 2.0).astype(np.float32)
        cy = np.where(n_vis > 0, ty_zero.sum(axis=1) / safe_n_vis, PITCH_WIDTH / 2.0).astype(np.float32)
        ms = np.where(n_spd > 0, ts_zero.sum(axis=1) / safe_n_spd, 0.0).astype(np.float32)

        var_x = np.where(valid_pos, (tx - cx[:, None]) ** 2, 0.0).sum(axis=1) / safe_n_vis
        var_y = np.where(valid_pos, (ty - cy[:, None]) ** 2, 0.0).sum(axis=1) / safe_n_vis
        std_x = np.sqrt(np.maximum(0.0, var_x)).astype(np.float32)
        std_y = np.sqrt(np.maximum(0.0, var_y)).astype(np.float32)

        feats[f"{prefix}_centroid_x"] = cx
        feats[f"{prefix}_centroid_y"] = cy
        feats[f"{prefix}_centroid_vx"] = np.gradient(cx, DT).astype(np.float32)
        feats[f"{prefix}_centroid_vy"] = np.gradient(cy, DT).astype(np.float32)
        feats[f"{prefix}_mean_speed"] = ms
        feats[f"{prefix}_length_x"] = (std_x * 2.0).astype(np.float32)
        feats[f"{prefix}_width_y"] = (std_y * 2.0).astype(np.float32)

        dx_to_ball = ball_x[:, None] - tx
        dy_to_ball = ball_y[:, None] - ty
        dist = np.where(valid_pos, np.hypot(dx_to_ball, dy_to_ball), 999.0).astype(np.float32)

        feats[f"{prefix}_in_3m"] = (dist <= CONTROL_RADIUS_3M).sum(axis=1).astype(np.int8)
        feats[f"{prefix}_in_5m"] = (dist <= SUPPORT_RADIUS_5M).sum(axis=1).astype(np.int8)

        # Time-to-Intercept (TTI) incorporating player velocity vector over REACTION_TIME_S (0.4s)
        proj_x = tx + tvx * REACTION_TIME_S
        proj_y = ty + tvy * REACTION_TIME_S
        tti = np.where(
            valid_pos,
            REACTION_TIME_S + np.hypot(ball_x[:, None] - proj_x, ball_y[:, None] - proj_y) / PLAYER_MAX_SPEED_MS,
            99.0,
        ).astype(np.float32)
        feats[f"{prefix}_min_tti_s"] = np.minimum(tti.min(axis=1), 20.0).astype(np.float32)
        pitch_ctrl_rates[prefix] = np.where(valid_pos, np.exp(-tti / PITCH_CONTROL_TAU_S), 0.0).sum(axis=1)

        # Fast O(M) top-3 nearest players via argpartition
        part_idx = np.argpartition(dist, kth=2, axis=1)[:, :3]
        top3_dist = np.take_along_axis(dist, part_idx, axis=1)
        top3_spd = np.take_along_axis(ts_zero, part_idx, axis=1)

        sub_order = np.argsort(top3_dist, axis=1)
        sorted_dist = np.take_along_axis(top3_dist, sub_order, axis=1)
        sorted_speed = np.take_along_axis(top3_spd, sub_order, axis=1)
        near1_slot = np.take_along_axis(part_idx, sub_order[:, :1], axis=1)

        for k in range(3):
            d_k = sorted_dist[:, k]
            s_k = np.where(d_k < 900.0, sorted_speed[:, k], 0.0)
            feats[f"{prefix}_near{k+1}_dist_m"] = np.minimum(d_k, 120.0).astype(np.float32)
            feats[f"{prefix}_near{k+1}_speed_ms"] = s_k.astype(np.float32)

        # Kinematics of the nearest player (near1)
        n1_vx = np.take_along_axis(tvx, near1_slot, axis=1)[:, 0]
        n1_vy = np.take_along_axis(tvy, near1_slot, axis=1)[:, 0]
        n1_dx = np.take_along_axis(np.where(valid_pos, dx_to_ball, 0.0), near1_slot, axis=1)[:, 0]
        n1_dy = np.take_along_axis(np.where(valid_pos, dy_to_ball, 0.0), near1_slot, axis=1)[:, 0]
        n1_d = feats[f"{prefix}_near1_dist_m"]

        rel_vx = n1_vx - ball_vx
        rel_vy = n1_vy - ball_vy
        closing_spd = np.where(n1_d < 110.0, (rel_vx * n1_dx + rel_vy * n1_dy) / (n1_d + 0.1), 0.0)
        feats[f"{prefix}_near1_closing_speed_ms"] = np.clip(closing_spd, -25.0, 25.0).astype(np.float32)

        n1_vnorm = np.hypot(n1_vx, n1_vy)
        vel_cos = np.where(
            (n1_vnorm > 0.3) & (ball_speed > 0.3),
            (n1_vx * ball_vx + n1_vy * ball_vy) / (n1_vnorm * ball_speed + 1e-5),
            0.0,
        )
        feats[f"{prefix}_near1_vel_cos"] = np.clip(vel_cos, -1.0, 1.0).astype(np.float32)

    diff_m = (feats["away_near1_dist_m"] - feats["home_near1_dist_m"]).astype(np.float32)
    feats["near1_dist_diff_m"] = diff_m
    feats["tti_diff_s"] = (feats["away_min_tti_s"] - feats["home_min_tti_s"]).astype(np.float32)

    tot_rate = pitch_ctrl_rates["home"] + pitch_ctrl_rates["away"]
    feats["home_pitch_control"] = np.where(tot_rate > 1e-6, pitch_ctrl_rates["home"] / tot_rate, 0.5).astype(np.float32)

    feats["is_contested"] = (
        (feats["home_near1_dist_m"] <= CONTESTED_RADIUS_M) &
        (feats["away_near1_dist_m"] <= CONTESTED_RADIUS_M)
    ).astype(np.int8)

    defl_angle = compute_ball_deflection_angle(ball_x, ball_y, status_code)
    feats["ball_deflection_angle"] = defl_angle

    touch_feats = compute_touch_and_control_proxies(
        feats["home_near1_dist_m"], feats["away_near1_dist_m"], defl_angle, status_code, period
    )
    feats.update(touch_feats)

    # Half-isolated rolling control windows
    roll50 = np.zeros(n_frames, dtype=np.float32)
    roll100 = np.zeros(n_frames, dtype=np.float32)
    raw_obs = np.where(status_code <= 1, diff_m, np.nan)

    for p_val in (1, 2):
        p_idx = np.flatnonzero(period == p_val)
        if len(p_idx) == 0:
            continue
        s_half = pd.Series(raw_obs[p_idx]).ffill().bfill().fillna(0.0)
        roll50[p_idx] = s_half.rolling(WINDOW_2S_FRAMES, min_periods=1, center=True).mean().astype(np.float32).to_numpy()
        roll100[p_idx] = s_half.rolling(WINDOW_4S_FRAMES, min_periods=1, center=True).mean().astype(np.float32).to_numpy()

    feats["control_diff_roll50"] = roll50
    feats["control_diff_roll100"] = roll100
    return feats


def verify_and_normalize_halves(feats, ball_x, ball_vx, period):
    """Adds direction-normalized X coordinates (*_norm) where HOME attacks towards +X in both halves."""
    h1_mask, h2_mask = (period == 1), (period == 2)
    h1_home_x = float(np.mean(feats["home_centroid_x"][h1_mask]))
    h1_away_x = float(np.mean(feats["away_centroid_x"][h1_mask]))
    h2_home_x = float(np.mean(feats["home_centroid_x"][h2_mask]))
    h2_away_x = float(np.mean(feats["away_centroid_x"][h2_mask]))

    flip_mask = (h1_mask & (h1_home_x > h1_away_x)) | (h2_mask & (h2_home_x > h2_away_x))
    feats["ball_x_norm"] = np.where(flip_mask, PITCH_LENGTH - ball_x, ball_x).astype(np.float32)
    feats["ball_vx_norm"] = np.where(flip_mask, -ball_vx, ball_vx).astype(np.float32)
    feats["home_centroid_x_norm"] = np.where(flip_mask, PITCH_LENGTH - feats["home_centroid_x"], feats["home_centroid_x"]).astype(np.float32)
    feats["away_centroid_x_norm"] = np.where(flip_mask, PITCH_LENGTH - feats["away_centroid_x"], feats["away_centroid_x"]).astype(np.float32)


# ---------------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------------
def build_stage1():
    t0 = time.time()
    meta_path, raw_path, st_path, n_frames, (h1_s, h1_e, h2_s, h2_e) = resolve_match_paths()

    slot_teams = get_slot_teams(meta_path)
    print(f"\n[1/5] Loading {n_frames} frames ({len(slot_teams)} slots) from {os.path.basename(raw_path)}...")
    px, py, ps, bx, by = parse_raw_atd(raw_path, n_frames, len(slot_teams))
    px, py, ps = heal_player_tracks(px, py, ps, slot_teams)
    pvx, pvy = compute_player_velocities(px, py)

    print("[2/5] Computing Focal Action Centroid, gap timers, and boundary-safe ball kinematics...")
    att_x, att_y, att_spread = compute_focal_attention_centroid(px, py, ps, pvx, pvy)
    ball_x, ball_y, status_code, status_label, sigma_m, gap_len, f_since, f_to, out = impute_ball_and_uncertainty(
        bx, by, att_x, att_y, att_spread
    )
    ball_vx, ball_vy, ball_speed, ball_acc = compute_clean_ball_kinematics(ball_x, ball_y, status_code)

    frames = np.arange(n_frames, dtype=np.int32)
    period = np.zeros(n_frames, dtype=np.int8)
    period[(frames >= h1_s) & (frames <= h1_e)] = 1
    period[(frames >= h2_s) & (frames <= h2_e)] = 2

    is_graded = (period > 0).astype(np.int8)
    is_pre_kickoff_d = ((frames >= h2_s) & (frames < KICKOFF_2ND_HALF)).astype(np.int8)

    print("[3/5] Extracting canonical team features, TTI, Pitch Control, and touch continuity proxies...")
    team_feats = extract_canonical_team_features(
        px, py, ps, pvx, pvy, slot_teams, ball_x, ball_y, ball_vx, ball_vy, ball_speed, status_code, period
    )
    verify_and_normalize_halves(team_feats, ball_x, ball_vx, period)

    print("[4/5] Integrating Smart Tagging priors (with DEFEND* polarity fix & Sideline N/A resolver)...")
    st_feats = resolve_smart_tagging_na_and_enhance(st_path, n_frames, px, py, slot_teams, ball_x, ball_y)

    df = pd.DataFrame({
        "frame_id": frames,
        "period": period,
        "is_graded": is_graded,
        "is_pre_kickoff_d": is_pre_kickoff_d,
        "timestamp_s": (frames * DT).astype(np.float32),
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_vx": ball_vx,
        "ball_vy": ball_vy,
        "ball_speed": ball_speed,
        "ball_acc": ball_acc,
        "ball_status": status_label,
        "ball_status_code": status_code,
        "ball_sigma_m": sigma_m,
        "ball_gap_frames": gap_len,
        "frames_since_ball_gap": f_since,
        "frames_to_ball_gap": f_to,
        "ball_near_boundary": out,
        **st_feats,
        **team_feats,
    })

    assert len(df) == n_frames, f"Length mismatch error: {len(df)} != {n_frames}"
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"\n[WARNING] Unexpected NaN values in columns: {nan_cols} — patching safely.")
        df[nan_cols] = df[nan_cols].ffill().bfill().fillna(0.0)
    assert not df.columns[df.isna().any()].tolist(), "NaN values still present!"

    out_path = os.path.join(ROOT, "data", "interim", "stage1_tracking.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"\n[5/5] SUCCESS! Saved {out_path} ({len(df.columns)} columns) in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    build_stage1()