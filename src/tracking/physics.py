"""
Physics & Kinematics module: Tracklet Healer, Savitzky-Golay filtering, Focal Action Centroid, and Ball Imputation.
"""

import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_coeffs

try:
    from .config import (
        PITCH_LENGTH, PITCH_WIDTH, DT, HOME, AWAY,
        CLONE_DIST_THRESHOLD_M, SAVGOL_WINDOW, SAVGOL_POLY,
        SHORT_GAP_MAX_FRAMES, MAX_BALL_SPEED_MS
    )
except ImportError:
    from config import (
        PITCH_LENGTH, PITCH_WIDTH, DT, HOME, AWAY,
        CLONE_DIST_THRESHOLD_M, SAVGOL_WINDOW, SAVGOL_POLY,
        SHORT_GAP_MAX_FRAMES, MAX_BALL_SPEED_MS
    )


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