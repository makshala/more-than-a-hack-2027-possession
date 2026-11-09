"""
Role 2 (Tracking): Generating the data contract data/interim/stage1_tracking.parquet (v3.7 Top-1% Tactical + Broadcast Fusion).
Main orchestrator script.
"""

import os
import time
import numpy as np
import pandas as pd

try:
    from .config import ROOT, DT, KICKOFF_2ND_HALF
    from .io_utils import (
        resolve_match_paths, get_slot_teams, parse_raw_atd,
        extract_homography_features, load_stage0_broadcast_features,
        audit_and_patch_nans
    )
    from .physics import (
        heal_player_tracks, compute_player_velocities_savgol,
        compute_focal_attention_centroid, impute_ball_and_uncertainty,
        compute_clean_ball_kinematics
    )
    from .tactics import (
        resolve_smart_tagging_na_and_enhance, extract_canonical_team_features,
        verify_and_normalize_halves_and_gk
    )
except ImportError:
    from config import ROOT, DT, KICKOFF_2ND_HALF
    from io_utils import (
        resolve_match_paths, get_slot_teams, parse_raw_atd,
        extract_homography_features, load_stage0_broadcast_features,
        audit_and_patch_nans
    )
    from physics import (
        heal_player_tracks, compute_player_velocities_savgol,
        compute_focal_attention_centroid, impute_ball_and_uncertainty,
        compute_clean_ball_kinematics
    )
    from tactics import (
        resolve_smart_tagging_na_and_enhance, extract_canonical_team_features,
        verify_and_normalize_halves_and_gk
    )


def build_stage1():
    t0 = time.time()
    meta_path, raw_path, st_path, homo_path, n_frames, (h1_s, h1_e, h2_s, h2_e) = resolve_match_paths()
    slot_teams = get_slot_teams(meta_path)

    frames, period = np.arange(n_frames, dtype=np.int32), np.zeros(n_frames, dtype=np.int8)
    period[(frames >= h1_s) & (frames <= h1_e)], period[(frames >= h2_s) & (frames <= h2_e)] = 1, 2
    is_graded = (period > 0).astype(np.int8)
    is_pre_kickoff_d = ((frames >= h2_s) & (frames < KICKOFF_2ND_HALF)).astype(np.int8)

    print(f"\n[1/5] Loading {n_frames} frames ({len(slot_teams)} slots) from {os.path.basename(raw_path)}...")
    px, py, ps, bx, by = parse_raw_atd(raw_path, n_frames, len(slot_teams))
    px, py, ps = heal_player_tracks(px, py, ps, slot_teams)
    px, py, ps, pvx, pvy = compute_player_velocities_savgol(px, py, ps)

    print("[2/5] Integrating Broadcast stage0, Focal Action Centroid & ball kinematics...")
    lookat_x, lookat_y, brd_bx, brd_by, brd_feats = load_stage0_broadcast_features(n_frames, period)
    att_x, att_y, att_spread = compute_focal_attention_centroid(px, py, ps, pvx, pvy, lookat_x, lookat_y)
    ball_x, ball_y, status_code, status_label, sigma_m, gap_len, f_since, f_to, out = impute_ball_and_uncertainty(
        bx, by, att_x, att_y, att_spread, brd_bx, brd_by
    )
    ball_vx, ball_vy, ball_speed, ball_acc = compute_clean_ball_kinematics(ball_x, ball_y, status_code)

    brd_feats["ball_to_brd_lookat_dist_m"] = np.hypot(
        ball_x - brd_feats["brd_lookat_x"], ball_y - brd_feats["brd_lookat_y"]
    ).astype(np.float32)

    print("[3/5] Extracting canonical team features, GK proxy, Dead-Ball heuristics, and dual rolling windows...")
    team_feats = extract_canonical_team_features(
        px, py, ps, pvx, pvy, slot_teams, ball_x, ball_y, ball_vx, ball_vy, ball_speed, status_code, period
    )
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
        **homo_feats, **brd_feats, **st_feats, **team_feats,
    })

    audit_and_patch_nans(df)
    out_path = os.path.join(ROOT, "data", "interim", "stage1_tracking.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n[5/5] SUCCESS! Saved {out_path} ({len(df.columns)} columns) in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    build_stage1()