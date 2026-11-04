"""
IO utilities: path resolution, raw ATD parsing, homography extraction, and NaN auditing.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

try:
    from .config import (
        ROOT, games, GAME, PITCH_LENGTH, PITCH_WIDTH, FPS, DT,
        H1_START, H1_END, H2_START, H2_END, HOME, AWAY
    )
except ImportError:
    from config import (
        ROOT, games, GAME, PITCH_LENGTH, PITCH_WIDTH, FPS, DT,
        H1_START, H1_END, H2_START, H2_END, HOME, AWAY
    )


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