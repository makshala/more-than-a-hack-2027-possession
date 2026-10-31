"""
Role 2 (Tracking): Generating the data contract data/interim/stage1_tracking.parquet (v2.2).

Audit Fixes & Optimizations over 2.1(2.0 and 2.1 wasn`t pushed to GitHub):
  1. Fixed Inter-slot stitching bug: copies only the contiguous segment [b_frame:b_end] instead of
     overwriting the entire match tail, and uses true Hungarian assignment (linear_sum_assignment).
  2. Eliminated root cause of speed NaNs / RuntimeWarnings: imputes missing slot speeds kinematically
     when (X, Y) are present but speed is NaN, and uses separate valid masks for position vs speed.
  3. Removed cross-half data leakage: rolling control windows and ffill/bfill operate strictly
     within each half independently.
  4. Added Spatial Clone Deduplication: suppresses duplicate tracker slots (< 0.45m apart within same team)
     so near2_dist_m and in_3m counts are never corrupted by ghost detections (DATA.md).
  5. Replaced O(M log M) argsort with O(M) np.argpartition for top-3 nearest player extraction.
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

# Domain-grounded physical constants
SHORT_GAP_MAX_FRAMES = 25
CONTESTED_RADIUS_M = 2.5
CONTROL_RADIUS_3M = 3.0
SUPPORT_RADIUS_5M = 5.0
CLONE_DIST_THRESHOLD_M = 0.45  # Two slots of same team closer than 45cm = tracker duplicate
WINDOW_2S_FRAMES = 50
WINDOW_4S_FRAMES = 100

# Official frame boundaries
H1_START, H1_END = 11185, 90651
H2_START, H2_END = 110808, 190197
KICKOFF_2ND_HALF = 113742

HOME = 0
AWAY = 1


def resolve_match_paths():
    """Resolves paths via games.asset_paths() if available, falling back to standard data/ layout."""
    if games is not None and hasattr(games, "asset_paths"):
        paths = games.asset_paths(game=GAME)
        meta_path = paths["atd_meta"]
        raw_path = paths["atd_raw"]
        st_path = paths.get("smart_tagging", os.path.join(ROOT, "data", "tactical", f"{GAME}_tactical_smart_tagging.xml"))
    else:
        base_dir = os.environ.get("MTAH_DATA_DIR", os.path.join(ROOT, "data"))
        tac_dir = os.path.join(base_dir, "tactical")
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
    """Maps each player slot by its ShirtNumber (1-based slot index) to avoid ordering bugs."""
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


def heal_player_tracks(px, py, ps, slot_teams, max_stitch_frames=25):
    """
    Heals identity handoffs and suppresses duplicate clone slots:
      1. Inter-slot Hungarian stitching (linear_sum_assignment) on contiguous tracklet segments.
      2. Spatial clone deduplication (< 0.45m within same team).
    """
    n_frames, n_slots = px.shape
    px_h, py_h, ps_h = px.copy(), py.copy(), ps.copy()
    total_stitched = 0
    total_clones_removed = 0

    for team_code in (HOME, AWAY):
        team_slots = np.flatnonzero(slot_teams == team_code)
        deaths = []  # (slot, end_frame, x, y, vx, vy)
        births = []  # (slot, start_frame, end_frame, x, y)

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
            INF = 1e6
            cost_mat = np.full((n_d, n_b), INF, dtype=np.float32)

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

                # Check that d_slot is actually empty across [d_f + 1 .. b_end]
                if not np.all(np.isnan(px_h[d_f + 1:b_end + 1, d_slot])):
                    continue

                gap_len = b_f - d_f
                alpha = np.linspace(0.0, 1.0, gap_len + 1, dtype=np.float32)[1:-1]
                if len(alpha) > 0:
                    px_h[d_f + 1:b_f, d_slot] = (1.0 - alpha) * dx + alpha * bx_val
                    py_h[d_f + 1:b_f, d_slot] = (1.0 - alpha) * dy + alpha * by_val
                    ps_h[d_f + 1:b_f, d_slot] = (1.0 - alpha) * np.hypot(dvx, dvy) + alpha * ps_h[b_f, b_slot]

                # Move ONLY the contiguous segment [b_f .. b_end]
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

        # Spatial clone deduplication: if >11 visible or two slots < 0.45m apart
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


def resolve_smart_tagging_na_and_enhance(st_path, n_frames, px, py, slot_teams, ball_x, ball_y):
    """Algorithmically resolves 'Team = N/A' using player geometry without future leakage."""
    st_set_piece = np.zeros(n_frames, dtype=np.int8)
    st_throw_in = np.zeros(n_frames, dtype=np.int8)
    home_votes = np.zeros(n_frames, dtype=np.float32)
    away_votes = np.zeros(n_frames, dtype=np.float32)
    st_n_active = np.zeros(n_frames, dtype=np.int8)

    if not os.path.exists(st_path):
        return {
            "st_set_piece": st_set_piece,
            "st_throw_in": st_throw_in,
            "st_team_prior": np.zeros(n_frames, dtype=np.int8),
            "st_n_active_tags": st_n_active,
        }

    home_mask = (slot_teams == HOME)
    away_mask = (slot_teams == AWAY)
    resolved_na_count = 0

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
        st_n_active[s_frame:e_frame] = np.minimum(120, st_n_active[s_frame:e_frame] + 1)

        if code == "SET PIECES":
            st_set_piece[s_frame:e_frame] = 1
            if "throw-in" in tags.get("type", "").lower():
                st_throw_in[s_frame:e_frame] = 1

        if "barcelona" in team_val:
            home_votes[s_frame:e_frame] += 1.0
        elif "atletic" in team_val or "athletic" in team_val:
            away_votes[s_frame:e_frame] += 1.0
        else:
            w_start, w_end = max(0, e_frame - 15), min(n_frames, e_frame + 25)
            if w_end > w_start:
                bx_w, by_w = ball_x[w_start:w_end, None], ball_y[w_start:w_end, None]
                h_d = np.hypot(px[w_start:w_end][:, home_mask] - bx_w, py[w_start:w_end][:, home_mask] - by_w)
                a_d = np.hypot(px[w_start:w_end][:, away_mask] - bx_w, py[w_start:w_end][:, away_mask] - by_w)

                h_med = float(np.nanmedian(np.where( np.all(np.isnan(h_d), axis=1), 999.0, np.nanmin(np.where(np.isnan(h_d), 999.0, h_d), axis=1))))
                a_med = float(np.nanmedian(np.where( np.all(np.isnan(a_d), axis=1), 999.0, np.nanmin(np.where(np.isnan(a_d), 999.0, a_d), axis=1))))

                # Vote applies to the interval and immediate 1s restart window (e_frame + 25)
                vote_end = min(n_frames, e_frame + 25)
                if h_med < a_med:
                    home_votes[s_frame:vote_end] += 0.75
                elif a_med < h_med:
                    away_votes[s_frame:vote_end] += 0.75
                resolved_na_count += 1

    print(f"[SMART TAGGING ENHANCER] Algorithmically resolved {resolved_na_count} 'Team=N/A' instances.")
    return {
        "st_set_piece": st_set_piece,
        "st_throw_in": st_throw_in,
        "st_team_prior": np.sign(home_votes - away_votes).astype(np.int8),
        "st_n_active_tags": st_n_active,
    }


def compute_attention_centroid(px, py, ps):
    """Estimates the Virtual Ball (Attention Centroid) weighted by log1p(speed)."""
    valid = ~np.isnan(px)
    n_visible = valid.sum(axis=1)

    weights = np.where(valid, 1.0 + np.log1p(np.maximum(0.0, np.nan_to_num(ps, nan=0.0))), 0.0)
    weight_sum = weights.sum(axis=1)

    safe_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    att_x = np.where(n_visible > 0, np.nansum(px * weights, axis=1) / safe_sum, PITCH_LENGTH / 2.0)
    att_y = np.where(n_visible > 0, np.nansum(py * weights, axis=1) / safe_sum, PITCH_WIDTH / 2.0)

    dx = np.where(valid, px - att_x[:, None], 0.0)
    dy = np.where(valid, py - att_y[:, None], 0.0)
    spread_m = np.sqrt(np.where(n_visible > 0, ((dx ** 2 + dy ** 2) * weights).sum(axis=1) / safe_sum, 100.0))

    return att_x.astype(np.float32), att_y.astype(np.float32), spread_m.astype(np.float32)


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
    """Classifies frames into OBSERVED (0), INTERPOLATED_SHORT (1), INFERRED_LATENT (2)."""
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
            blend_len = min(25, gap_len)

            if has_left:
                w = np.linspace(0.8, 0.0, blend_len, dtype=np.float32)
                est_x[:blend_len] = w * bx[start - 1] + (1.0 - w) * est_x[:blend_len]
                est_y[:blend_len] = w * by[start - 1] + (1.0 - w) * est_y[:blend_len]
            if has_right:
                w = np.linspace(0.0, 0.8, blend_len, dtype=np.float32)
                est_x[-blend_len:] = w * bx[end] + (1.0 - w) * est_x[-blend_len:]
                est_y[-blend_len:] = w * by[end] + (1.0 - w) * est_y[-blend_len:]

            ball_x[start:end] = est_x
            ball_y[start:end] = est_y
            sigma_m[start:end] = np.clip(0.35 * att_spread[start:end], 3.5, 25.0)

    status_label = np.array(["OBSERVED", "INTERPOLATED_SHORT", "INFERRED_LATENT"])[status_code]
    frames_since_gap, frames_to_gap = compute_gap_boundary_timers(status_code)
    out_of_bounds = (
        (ball_x < 0.5) | (ball_x > PITCH_LENGTH - 0.5) |
        (ball_y < 0.5) | (ball_y > PITCH_WIDTH - 0.5)
    ).astype(np.int8)

    return ball_x, ball_y, status_code, status_label, sigma_m, gap_length_arr, frames_since_gap, frames_to_gap, out_of_bounds


def extract_canonical_team_features(px, py, ps, slot_teams, ball_x, ball_y, status_code, period):
    """
    Transforms jumping slots into permutation-invariant team features using O(M) np.argpartition
    and strictly isolates rolling windows per half to prevent halftime leakage.
    """
    feats = {}
    n_frames = px.shape[0]

    for team_code, prefix in ((HOME, "home"), (AWAY, "away")):
        mask = (slot_teams == team_code)
        tx, ty, ts = px[:, mask], py[:, mask], ps[:, mask]

        valid_pos = ~np.isnan(tx)
        valid_spd = ~np.isnan(ts)
        n_vis = valid_pos.sum(axis=1).astype(np.int8)
        n_spd = valid_spd.sum(axis=1)
        feats[f"{prefix}_n_visible"] = n_vis

        # Vectorized mean/std without empty-slice RuntimeWarnings
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

        dx, dy = tx - ball_x[:, None], ty - ball_y[:, None]
        dist = np.where(valid_pos, np.hypot(dx, dy), 999.0).astype(np.float32)

        feats[f"{prefix}_in_3m"] = (dist <= CONTROL_RADIUS_3M).sum(axis=1).astype(np.int8)
        feats[f"{prefix}_in_5m"] = (dist <= SUPPORT_RADIUS_5M).sum(axis=1).astype(np.int8)

        # Fast O(M) top-3 selection via argpartition + 3-column sort
        part_idx = np.argpartition(dist, kth=2, axis=1)[:, :3]
        top3_dist = np.take_along_axis(dist, part_idx, axis=1)
        top3_spd = np.take_along_axis(ts_zero, part_idx, axis=1)

        sub_order = np.argsort(top3_dist, axis=1)
        sorted_dist = np.take_along_axis(top3_dist, sub_order, axis=1)
        sorted_speed = np.take_along_axis(top3_spd, sub_order, axis=1)

        for k in range(3):
            d_k = sorted_dist[:, k]
            s_k = np.where(d_k < 900.0, sorted_speed[:, k], 0.0)
            feats[f"{prefix}_near{k+1}_dist_m"] = np.minimum(d_k, 120.0).astype(np.float32)
            feats[f"{prefix}_near{k+1}_speed_ms"] = s_k.astype(np.float32)

    diff_m = (feats["away_near1_dist_m"] - feats["home_near1_dist_m"]).astype(np.float32)
    feats["near1_dist_diff_m"] = diff_m
    feats["is_contested"] = (
        (feats["home_near1_dist_m"] <= CONTESTED_RADIUS_M) &
        (feats["away_near1_dist_m"] <= CONTESTED_RADIUS_M)
    ).astype(np.int8)

    # Compute rolling control windows strictly per half to prevent halftime bleed
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
# 7. MAIN PIPELINE
# ---------------------------------------------------------
def build_stage1():
    t0 = time.time()
    meta_path, raw_path, st_path, n_frames, (h1_s, h1_e, h2_s, h2_e) = resolve_match_paths()

    slot_teams = get_slot_teams(meta_path)
    print(f"\n[1/5] Loading {n_frames} frames ({len(slot_teams)} slots) from {os.path.basename(raw_path)}...")
    px, py, ps, bx, by = parse_raw_atd(raw_path, n_frames, len(slot_teams))
    px, py, ps = heal_player_tracks(px, py, ps, slot_teams)

    print("[2/5] Computing Attention Centroid, gap timers, and imputing missing ball data...")
    att_x, att_y, att_spread = compute_attention_centroid(px, py, ps)
    ball_x, ball_y, status_code, status_label, sigma_m, gap_len, f_since, f_to, out = impute_ball_and_uncertainty(
        bx, by, att_x, att_y, att_spread
    )

    ball_vx = np.gradient(ball_x, DT).astype(np.float32)
    ball_vy = np.gradient(ball_y, DT).astype(np.float32)
    ball_speed = np.hypot(ball_vx, ball_vy).astype(np.float32)
    ball_acc = np.hypot(np.gradient(ball_vx, DT), np.gradient(ball_vy, DT)).astype(np.float32)

    frames = np.arange(n_frames, dtype=np.int32)
    period = np.zeros(n_frames, dtype=np.int8)
    period[(frames >= h1_s) & (frames <= h1_e)] = 1
    period[(frames >= h2_s) & (frames <= h2_e)] = 2

    is_graded = (period > 0).astype(np.int8)
    is_pre_kickoff_d = ((frames >= h2_s) & (frames < KICKOFF_2ND_HALF)).astype(np.int8)

    print("[3/5] Extracting canonical team, duel, and rolling control features...")
    team_feats = extract_canonical_team_features(px, py, ps, slot_teams, ball_x, ball_y, status_code, period)
    verify_and_normalize_halves(team_feats, ball_x, ball_vx, period)

    print("[4/5] Integrating Smart Tagging priors and resolving N/A cases...")
    st_feats = resolve_smart_tagging_na_and_enhance(st_path, n_frames, px, py, slot_teams, ball_x, ball_y)

    df = pd.DataFrame({
        "frame_id": frames, "period": period, "is_graded": is_graded, "is_pre_kickoff_d": is_pre_kickoff_d,
        "timestamp_s": (frames * DT).astype(np.float32),
        "ball_x": ball_x, "ball_y": ball_y, "ball_vx": ball_vx, "ball_vy": ball_vy,
        "ball_speed": ball_speed, "ball_acc": ball_acc,
        "ball_status": status_label, "ball_status_code": status_code, "ball_sigma_m": sigma_m,
        "ball_gap_frames": gap_len, "frames_since_ball_gap": f_since, "frames_to_ball_gap": f_to,
        "ball_near_boundary": out,
        **st_feats, **team_feats,
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