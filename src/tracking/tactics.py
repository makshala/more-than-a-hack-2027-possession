"""
Tactics & Domain Features module: Smart Tagging decomposition, Touch/Control state machine,
Pitch Control, TTI, Dual Rolling Windows, and Half/Goalkeeper normalization.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.signal import lfilter

try:
    from .config import (
        PITCH_LENGTH, PITCH_WIDTH, FPS, DT, HOME, AWAY,
        CONTESTED_RADIUS_M, CONTROL_RADIUS_3M, SUPPORT_RADIUS_5M,
        WINDOW_2S_FRAMES, WINDOW_4S_FRAMES, REACTION_TIME_S,
        PLAYER_MAX_SPEED_MS, PITCH_CONTROL_TAU_S
    )
except ImportError:
    from config import (
        PITCH_LENGTH, PITCH_WIDTH, FPS, DT, HOME, AWAY,
        CONTESTED_RADIUS_M, CONTROL_RADIUS_3M, SUPPORT_RADIUS_5M,
        WINDOW_2S_FRAMES, WINDOW_4S_FRAMES, REACTION_TIME_S,
        PLAYER_MAX_SPEED_MS, PITCH_CONTROL_TAU_S
    )


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
                    h_side_dist = 0.7 * float(np.nanmedian(h_near1[w_start:w_end])) + 0.3 * min(h_y_med, PITCH_WIDTH - h_y_med)
                    a_side_dist = 0.7 * float(np.nanmedian(a_near1[w_start:w_end])) + 0.3 * min(a_y_med, PITCH_WIDTH - a_y_med)
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