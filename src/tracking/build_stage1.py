"""
Role 2 (Tracking): Generating the data contract data/interim/stage1_tracking.parquet.
Solves the problem of 46 jumping slots and 41% missing ball frames.
Provides OBSERVED, INTERPOLATED_SHORT, and INFERRED_LATENT ball statuses.
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. HARDCODED CONSTANTS AND PARAMETERS (Instead of games.py)
# ---------------------------------------------------------
ROOT = os.path.abspath(os.getcwd())  # Assuming execution from the root folder
GAME = "BAR-ATH"
PITCH_LENGTH = 105.3  # in meters
PITCH_WIDTH = 68.0    # in meters
FPS = 25.0
DT = 1.0 / FPS
SHORT_GAP_MAX_FRAMES = 25  # 1 second (maximum duration of a short gap)

# Time boundaries for the match
KICKOFF_1ST_HALF = 11185
HALFTIME_APPROX = 100000
KICKOFF_2ND_HALF = 113742
LAST_GRADED_FRAME = 190197

HOME = 0
AWAY = 1


# ---------------------------------------------------------
# 2. HELPER AND COMPUTATIONAL FUNCTIONS
# ---------------------------------------------------------
def get_slot_teams(meta_path):
    """Returns an array assigning each slot to a team: 0 = HOME, 1 = AWAY."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)


def parse_raw_atd(raw_path, n_frames, n_slots):
    """Fast streaming parser for ATD (without kloppy). Returns arrays in meters."""
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path, "r") as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frame_idx = int(parts[0]) - 1

            # Parse players (46 slots: x, y, speed)
            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                c1 = group.find(",")
                c2 = group.find(",", c1 + 1)
                px[frame_idx, slot] = float(group[:c1]) * PITCH_LENGTH
                py[frame_idx, slot] = float(group[c1 + 1:c2]) * PITCH_WIDTH
                ps[frame_idx, slot] = float(group[c2 + 1:])

            # Parse ball (x, y, speed)
            ball = parts[2]
            if ball and ball[0] != "N":
                c1 = ball.find(",")
                c2 = ball.find(",", c1 + 1)
                bx[frame_idx] = float(ball[:c1]) * PITCH_LENGTH
                by[frame_idx] = float(ball[c1 + 1:c2]) * PITCH_WIDTH

    return px, py, ps, bx, by


def compute_attention_centroid(px, py, ps):
    """
    Estimates the 'Virtual Ball' (Attention Centroid) using player positions and speeds.
    Running players have a significantly higher weight than standing players.
    """
    valid = ~np.isnan(px)
    n_visible = valid.sum(axis=1)

    # Weight formula: (1.0 + speed^2)
    weights = np.where(valid, 1.0 + np.nan_to_num(ps, nan=0.0) ** 2, 0.0)
    weight_sum = weights.sum(axis=1)

    safe_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    att_x = np.where(n_visible > 0, np.nansum(px * weights, axis=1) / safe_sum, PITCH_LENGTH / 2.0)
    att_y = np.where(n_visible > 0, np.nansum(py * weights, axis=1) / safe_sum, PITCH_WIDTH / 2.0)

    # Calculate spatial spread (standard deviation) of players around the centroid
    dx = np.where(valid, px - att_x[:, None], 0.0)
    dy = np.where(valid, py - att_y[:, None], 0.0)
    dist_sq = dx ** 2 + dy ** 2
    spread_m = np.sqrt(np.where(n_visible > 0, (dist_sq * weights).sum(axis=1) / safe_sum, 100.0))

    return att_x.astype(np.float32), att_y.astype(np.float32), spread_m.astype(np.float32)


def impute_ball_and_uncertainty(bx, by, att_x, att_y, att_spread):
    """
    Classifies frames into OBSERVED (0), INTERPOLATED_SHORT (1), INFERRED_LATENT (2).
    Generates continuous (ball_x, ball_y) and the uncertainty radius (ball_sigma_m).
    """
    n_frames = len(bx)
    observed = ~np.isnan(bx)

    ball_x = bx.copy()
    ball_y = by.copy()
    status_code = np.full(n_frames, 2, dtype=np.int8)
    status_code[observed] = 0
    sigma_m = np.full(n_frames, 8.0, dtype=np.float32)
    sigma_m[observed] = 0.5

    padded = np.concatenate(([True], observed, [True]))
    gap_starts = np.flatnonzero(padded[:-1] & ~padded[1:])
    gap_ends = np.flatnonzero(~padded[:-1] & padded[1:])

    for start, end in zip(gap_starts, gap_ends):
        gap_len = end - start
        has_left = start > 0
        has_right = end < n_frames

        if gap_len <= SHORT_GAP_MAX_FRAMES and has_left and has_right:
            # Short gap: Linear interpolation
            status_code[start:end] = 1
            alpha = np.linspace(0.0, 1.0, gap_len + 2, dtype=np.float32)[1:-1]
            ball_x[start:end] = (1.0 - alpha) * bx[start - 1] + alpha * bx[end]
            ball_y[start:end] = (1.0 - alpha) * by[start - 1] + alpha * by[end]

            # Uncertainty increases towards the center of the gap
            dist_to_edge = np.minimum(np.arange(1, gap_len + 1), np.arange(gap_len, 0, -1))
            sigma_m[start:end] = 0.5 + 0.16 * dist_to_edge
        else:
            # Long gap: Use Attention Centroid with edge blending
            status_code[start:end] = 2
            est_x = att_x[start:end].copy()
            est_y = att_y[start:end].copy()

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
            
            # Uncertainty proportional to player spread
            sigma_m[start:end] = np.clip(0.35 * att_spread[start:end], 3.5, 25.0)

    status_label = np.array(["OBSERVED", "INTERPOLATED_SHORT", "INFERRED_LATENT"])[status_code]
    return ball_x, ball_y, status_code, status_label, sigma_m


def extract_canonical_team_features(px, py, ps, slot_teams, ball_x, ball_y):
    """
    Transforms 46 jumping slots into stable, permutation-invariant columns:
    - Formation shape (centroid, width, length)
    - Top 3 nearest players to the ball (distance and speed)
    """
    feats = {}
    for team_code, prefix in ((HOME, "home"), (AWAY, "away")):
        mask = (slot_teams == team_code)
        tx = px[:, mask]
        ty = py[:, mask]
        ts = ps[:, mask]

        valid = ~np.isnan(tx)
        n_vis = valid.sum(axis=1).astype(np.int8)
        feats[f"{prefix}_n_visible"] = n_vis

        with np.errstate(invalid="ignore"):
            cx = np.nanmean(tx, axis=1)
            cy = np.nanmean(ty, axis=1)
            ms = np.nanmean(ts, axis=1)
            std_x = np.nanstd(tx, axis=1)
            std_y = np.nanstd(ty, axis=1)

        feats[f"{prefix}_centroid_x"] = np.nan_to_num(cx, nan=PITCH_LENGTH / 2.0).astype(np.float32)
        feats[f"{prefix}_centroid_y"] = np.nan_to_num(cy, nan=PITCH_WIDTH / 2.0).astype(np.float32)
        feats[f"{prefix}_mean_speed"] = np.nan_to_num(ms, nan=0.0).astype(np.float32)
        feats[f"{prefix}_length_x"] = np.nan_to_num(std_x * 2.0, nan=0.0).astype(np.float32)
        feats[f"{prefix}_width_y"] = np.nan_to_num(std_y * 2.0, nan=0.0).astype(np.float32)

        # Distances to the ball
        dx = tx - ball_x[:, None]
        dy = ty - ball_y[:, None]
        dist = np.where(valid, np.hypot(dx, dy), 999.0).astype(np.float32)

        # Sort players by distance to ball to ensure ID invariance
        order = np.argsort(dist, axis=1)
        sorted_dist = np.take_along_axis(dist, order, axis=1)
        sorted_speed = np.take_along_axis(np.nan_to_num(ts, nan=0.0), order, axis=1)

        for k in range(3):
            d_k = sorted_dist[:, k]
            s_k = np.where(d_k < 900.0, sorted_speed[:, k], 0.0)
            feats[f"{prefix}_near{k+1}_dist_m"] = np.minimum(d_k, 120.0).astype(np.float32)
            feats[f"{prefix}_near{k+1}_speed_ms"] = s_k.astype(np.float32)

    return feats


# ---------------------------------------------------------
# 3. MAIN PIPELINE
# ---------------------------------------------------------
def build_stage1():
    t0 = time.time()
    
    n_frames = 199319
    meta_path = os.path.join(ROOT, "data", "tactical", f"{GAME}_tactical_atd.xml")
    raw_path = os.path.join(ROOT, "data", "tactical", f"{GAME}_tactical_atd.txt")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing XML metadata file at path: {meta_path}")
        
    slot_teams = get_slot_teams(meta_path)
    n_slots = len(slot_teams)

    print(f"[1/4] Loading {n_frames} frames ({n_slots} slots) from {os.path.basename(raw_path)}...")
    px, py, ps, bx, by = parse_raw_atd(raw_path, n_frames, n_slots)

    print("[2/4] Computing Attention Centroid and imputing 41% missing ball data...")
    att_x, att_y, att_spread = compute_attention_centroid(px, py, ps)
    ball_x, ball_y, status_code, status_label, sigma_m = impute_ball_and_uncertainty(
        bx, by, att_x, att_y, att_spread
    )

    # Ball velocity and acceleration
    ball_vx = np.gradient(ball_x, DT).astype(np.float32)
    ball_vy = np.gradient(ball_y, DT).astype(np.float32)
    ball_speed = np.hypot(ball_vx, ball_vy).astype(np.float32)
    ball_acc = np.hypot(np.gradient(ball_vx, DT), np.gradient(ball_vy, DT)).astype(np.float32)

    print("[3/4] Extracting canonical features for HOME and AWAY teams...")
    team_feats = extract_canonical_team_features(px, py, ps, slot_teams, ball_x, ball_y)

    # Mark periods (0 = break/off-play, 1 = 1st half, 2 = 2nd half)
    frames = np.arange(n_frames, dtype=np.int32)
    period = np.zeros(n_frames, dtype=np.int8)
    period[(frames >= KICKOFF_1ST_HALF) & (frames < HALFTIME_APPROX)] = 1
    period[(frames >= KICKOFF_2ND_HALF) & (frames <= LAST_GRADED_FRAME)] = 2

    df = pd.DataFrame({
        "frame_id": frames,
        "period": period,
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
        **team_feats,
    })

    assert len(df) == n_frames, f"Length mismatch error: {len(df)} != {n_frames}"
    
    out_dir = os.path.join(ROOT, "data", "interim")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stage1_tracking.parquet")
    df.to_parquet(out_path, index=False)

    counts = pd.Series(status_label).value_counts()
    print(f"\n[4/4] SUCCESS! Saved {out_path} in {time.time() - t0:.1f}s.")
    print("Distribution of ball_status across the entire match:")
    for label in ("OBSERVED", "INTERPOLATED_SHORT", "INFERRED_LATENT"):
        c = counts.get(label, 0)
        print(f"  - {label:<20}: {c:>7} frames ({100.0 * c / n_frames:5.1f}%)")


if __name__ == "__main__":
    build_stage1()