# src/deadball/kinematic_d.py
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

N_FRAMES: int = 199_319
PITCH_LENGTH: float = 105.0
PITCH_WIDTH: float = 68.0
HALF_LENGTH: float = PITCH_LENGTH / 2.0  # 52.5 m
HALF_WIDTH: float = PITCH_WIDTH / 2.0    # 34.0 m


def _extract_slot_arrays_from_raw_atd(
    atd_df: pd.DataFrame,
    home_slots: Optional[List[int]] = None,
    away_slots: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wyodrębnia macierze (N, n_slots) dla pozycji X, Y (przeskalowanych do [-52.5, 52.5] x [-34, 34])
    oraz prędkości S (m/s) z 46-slotowego DataFrame'u ATD (np. z kloppy.metrica).
    """
    cols = atd_df.columns

    # Identyfikacja kolumn slotów (konwencja kloppy: '1_x', '1_y', '1_s' lub 'home_1_x' itp.)
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
            s_col = f"{base}_s" if f"{base}_s" in cols else (f"{base}_speed" if f"{base}_speed" in cols else None)
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
            # W standardowym 46-slotowym ATD pierwsza połowa slotów należy do Home, druga do Away
            if slot_num <= 23:
                h_indices.append(idx)
            else:
                a_indices.append(idx)

    x_norm = atd_df[slot_x_cols].to_numpy(dtype=np.float32)
    y_norm = atd_df[slot_y_cols].to_numpy(dtype=np.float32)

    # Sprawdzenie czy współrzędne są w [0, 1] czy już w metrach
    max_x = np.nanmax(x_norm) if x_norm.size > 0 else 1.0
    if max_x <= 1.5:
        x_m = (x_norm - 0.5) * PITCH_LENGTH
        y_m = (y_norm - 0.5) * PITCH_WIDTH
    else:
        x_m = x_norm - (PITCH_LENGTH / 2.0)
        y_m = y_norm - (PITCH_WIDTH / 2.0)

    # Prędkości graczy w m/s
    if all(sc is not None for sc in slot_s_cols) and len(slot_s_cols) > 0:
        speeds = atd_df[slot_s_cols].to_numpy(dtype=np.float32)
    else:
        # Wyznaczenie prędkości z różnic skończonych jeśli brak kolumn _s
        vx = np.gradient(x_m, 0.04, axis=0)
        vy = np.gradient(y_m, 0.04, axis=0)
        speeds = np.hypot(vx, vy).astype(np.float32)

    # Pozycja piłki przeskalowana do [-52.5, 52.5] x [-34, 34]
    bx_raw = atd_df["ball_x"].to_numpy(dtype=np.float32)
    by_raw = atd_df["ball_y"].to_numpy(dtype=np.float32)
    if np.nanmax(bx_raw) <= 1.5:
        ball_x = (bx_raw - 0.5) * PITCH_LENGTH
        ball_y = (by_raw - 0.5) * PITCH_WIDTH
    else:
        # W stage1_tracking.parquet zakres to [0, 105.3] x [0, 68.0]
        ball_x = bx_raw * (PITCH_LENGTH / 105.3) - HALF_LENGTH
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
    """
    Oblicza średnią odległość pomiędzy wszystkimi widocznymi graczami Home (H) i Away (A)
    dla każdej klatki w sposób w pełni zwektoryzowany w blokach (chunkach).
    """
    n_frames = hx.shape[0]
    mean_dist = np.full(n_frames, np.nan, dtype=np.float32)

    for start in range(0, n_frames, chunk_size):
        end = min(n_frames, start + chunk_size)
        # Wymiary: (chunk, n_H, 1) oraz (chunk, 1, n_A) -> (chunk, n_H, n_A)
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

    # Uzupełnienie ewentualnych pustych klatek wartością mediany
    nan_mask = np.isnan(mean_dist)
    if np.any(nan_mask):
        median_val = float(np.nanmedian(mean_dist)) if not np.all(nan_mask) else 22.0
        mean_dist[nan_mask] = median_val

    return mean_dist


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
    """
    Wykrywa uformowania struktur statycznych na boisku (wektoryzacja po macierzach 2D):
    1. Mur przed rzutem wolnym (free_kick_wall_flag): >= 3 zawodników jednej drużyny
       w odległości [8.0m, 11.5m] od wolno poruszającej się piłki, skupionych blisko siebie
       i stojących niemal w miejscu (średnia prędkość < 1.5 m/s).
    2. Zgrupowanie w polu karnym przed rożnym (corner_box_cluster_flag): >= 6 graczy
       w jednym polu karnym (|x| >= 36.0m, |y| <= 20.16m) przy niskiej prędkości graczy.
    """
    all_x = np.concatenate([hx, ax], axis=1)
    all_y = np.concatenate([hy, ay], axis=1)
    all_s = np.concatenate([hs, as_], axis=1)

    # 1. Zgrupowanie w polu karnym (pole karne: 16.5m od bramki -> |x| >= 52.5 - 16.5 = 36.0m, |y| <= 20.16m)
    in_left_box = (all_x <= -36.0) & (np.abs(all_y) <= 20.16)
    in_right_box = (all_x >= 36.0) & (np.abs(all_y) <= 20.16)
    n_in_box = np.maximum(np.sum(in_left_box, axis=1), np.sum(in_right_box, axis=1))

    mean_speed = np.nanmean(all_s, axis=1)
    mean_speed = np.nan_to_num(mean_speed, nan=0.0)
    corner_box_cluster = (n_in_box >= 6) & (mean_speed < 2.2)

    # 2. Mur przed rzutem wolnym: gracze w pierścieniu 8.0 - 11.5m od piłki o prędkości < 1.5 m/s
    dist_h_ball = np.hypot(hx - ball_x[:, None], hy - ball_y[:, None])
    dist_a_ball = np.hypot(ax - ball_x[:, None], ay - ball_y[:, None])

    h_in_wall_ring = (dist_h_ball >= 8.0) & (dist_h_ball <= 11.5) & (hs < 1.5)
    a_in_wall_ring = (dist_a_ball >= 8.0) & (dist_a_ball <= 11.5) & (as_ < 1.5)

    # Sprawdzenie zwartości kątowej/przestrzennej graczy w pierścieniu (odchylenie std pozycji Y < 2.5m)
    hy_wall = np.where(h_in_wall_ring, hy, np.nan)
    ay_wall = np.where(a_in_wall_ring, ay, np.nan)
    h_wall_compact = np.nanstd(hy_wall, axis=1) < 2.8
    a_wall_compact = np.nanstd(ay_wall, axis=1) < 2.8

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
    Agreguje wskaźniki braku gry (stanu D) na podstawie danych śledzenia.
    Obsługuje bezpośrednio 46-slotowy format ATD (raw_atd_df) oraz wzbogacony format
    stage1_tracking.parquet (stage1_df).
    """
    if stage1_df is None and raw_atd_df is None:
        raise ValueError("Należy przekazać przynajmniej jeden z argumentów: stage1_df lub raw_atd_df.")

    out = pd.DataFrame({"frame": np.arange(N_FRAMES, dtype=np.int32)})

    # Ścieżka A: Obliczenia bezpośrednio na 46 slotach śledzenia (jeśli przekazano raw_atd_df)
    if raw_atd_df is not None:
        hx, hy, hs, ax, ay, as_, ball_x, ball_y = _extract_slot_arrays_from_raw_atd(raw_atd_df)
        all_speeds = np.concatenate([hs, as_], axis=1)
        valid_mask = ~np.isnan(all_speeds)
        n_visible = np.sum(valid_mask, axis=1)

        # 1. Średnia prędkość wszystkich widocznych zawodników
        speed_sum = np.nansum(all_speeds, axis=1)
        mean_speed_all = np.where(n_visible > 0, speed_sum / np.maximum(n_visible, 1), 0.0).astype(np.float32)

        # 2. Odsetek zawodników poruszających się pieszo (prędkość < 2.0 m/s)
        walking_count = np.sum((all_speeds < 2.0) & valid_mask, axis=1)
        walking_ratio_2ms = np.where(n_visible > 0, walking_count / np.maximum(n_visible, 1), 1.0).astype(np.float32)

        # 3. Brak pressingu mierzony jako średnia odległość między graczami H a A
        no_pressing_dist_m = _compute_mean_interteam_distance_chunked(hx, hy, ax, ay)

        # 4. Flaga wyjścia piłki poza boisko: [-52.5, 52.5] x [-34, 34]
        ball_out_of_bounds = (
            (np.abs(ball_x) > HALF_LENGTH) | (np.abs(ball_y) > HALF_WIDTH)
        ) & (~np.isnan(ball_x))

        # 5. Wykrycie uformowania struktur statycznych (mur / rożny w polu karnym)
        fk_wall, corner_cluster = _detect_static_structures_from_slots(
            hx, hy, hs, ax, ay, as_, ball_x, ball_y
        )

    # Ścieżka B: Wektoryzacja na bazie kolumn z pliku stage1_tracking.parquet
    else:
        df = stage1_df
        # 1. Średnia prędkość wszystkich widocznych zawodników
        mean_speed_all = df["all_players_mean_speed"].to_numpy(dtype=np.float32)

        # 2. Odsetek zawodników poruszających się pieszo (< 2.0 m/s)
        # W stage1_tracking 'players_walking_ratio' jest zdefiniowane dla < 1.5 m/s;
        # skalujemy rozkład z uwzględnieniem średniej prędkości drużyn do progu < 2.0 m/s
        walk_15 = df["players_walking_ratio"].to_numpy(dtype=np.float32)
        speed_factor = np.clip((2.0 - mean_speed_all) * 0.15, 0.0, 0.25)
        walking_ratio_2ms = np.clip(walk_15 + speed_factor, 0.0, 1.0).astype(np.float32)

        # 3. Brak pressingu: estymacja średniej odległości H-A ze środków ciężkości i rozpiętości formacji
        dx_c = df["home_centroid_x"].to_numpy(dtype=np.float32) - df["away_centroid_x"].to_numpy(dtype=np.float32)
        dy_c = df["home_centroid_y"].to_numpy(dtype=np.float32) - df["away_centroid_y"].to_numpy(dtype=np.float32)
        centroid_dist = np.hypot(dx_c, dy_c)
        # Uwzględnienie wariancji przestrzennej formacji (length_x = 2*std_x, width_y = 2*std_y)
        var_x = (df["home_length_x"].to_numpy(dtype=np.float32) / 2.0) ** 2 + (
            df["away_length_x"].to_numpy(dtype=np.float32) / 2.0
        ) ** 2
        var_y = (df["home_width_y"].to_numpy(dtype=np.float32) / 2.0) ** 2 + (
            df["away_width_y"].to_numpy(dtype=np.float32) / 2.0
        ) ** 2
        # Zależność E[||X - Y||] ≈ sqrt(||mu_H - mu_A||^2 + Tr(Sigma_H + Sigma_A))
        no_pressing_dist_m = np.sqrt(centroid_dist**2 + var_x + var_y).astype(np.float32)

        # 4. Flaga wyjścia piłki poza boisko po przeskalowaniu do [-52.5, 52.5] x [-34, 34]
        # W stage1 ball_x jest w [0, 105.3], ball_y w [0, 68.0]
        bx_norm = df["ball_x"].to_numpy(dtype=np.float32) / 105.3
        by_norm = df["ball_y"].to_numpy(dtype=np.float32) / 68.0
        ball_x_centered = (bx_norm - 0.5) * PITCH_LENGTH
        ball_y_centered = (by_norm - 0.5) * PITCH_WIDTH

        oob_geom = (np.abs(ball_x_centered) >= HALF_LENGTH - 0.05) | (
            np.abs(ball_y_centered) >= HALF_WIDTH - 0.05
        )
        near_bound = df["ball_near_boundary"].to_numpy(dtype=np.int8) > 0
        ball_out_of_bounds = oob_geom | (near_bound & (df["ball_speed"].to_numpy(dtype=np.float32) < 1.5))

        # 5. Wykrycie uformowania struktur statycznych (mur przed wolnym / zgrupowanie przed rożnym)
        in_box = df["players_in_penalty_box"].to_numpy(dtype=np.int8)
        ball_stat = df["ball_stationary_frames"].to_numpy(dtype=np.int16)

        corner_cluster = (in_box >= 6) & (ball_stat >= 10) & (walking_ratio_2ms >= 0.60)

        # Mur przed rzutem wolnym: piłka nieruchoma (>15 klatek), wysoki odsetek stojących graczy,
        # brak bezpośredniej walki o piłkę (is_contested == 0), a 2. i 3. gracz w pobliżu ~9-11m
        not_contested = df["is_contested"].to_numpy(dtype=np.int8) == 0
        h_n2 = df["home_near2_dist_m"].to_numpy(dtype=np.float32)
        a_n2 = df["away_near2_dist_m"].to_numpy(dtype=np.float32)
        wall_dist_proxy = ((h_n2 >= 7.5) & (h_n2 <= 12.0)) | ((a_n2 >= 7.5) & (a_n2 <= 12.0))
        fk_wall = (ball_stat >= 15) & not_contested & wall_dist_proxy & (walking_ratio_2ms >= 0.65)

    out["mean_speed_all"] = mean_speed_all
    out["walking_ratio_2ms"] = walking_ratio_2ms
    out["no_pressing_dist_m"] = no_pressing_dist_m
    out["ball_out_of_bounds"] = ball_out_of_bounds.astype(bool)
    out["free_kick_wall_flag"] = fk_wall.astype(bool)
    out["corner_box_cluster_flag"] = corner_cluster.astype(bool)
    out["static_structure_flag"] = (fk_wall | corner_cluster).astype(bool)

    # Syntetyczny wynik kinematyczny stanu D (kinematic_d_score w [0, 1])
    s_speed = np.clip((3.2 - mean_speed_all) / 2.0, 0.0, 1.0)
    s_walk = np.clip((walking_ratio_2ms - 0.35) / 0.50, 0.0, 1.0)
    # Brak pressingu w okolicach wznowień: duża separacja lub brak bezpośredniej walki o piłkę
    s_press = np.clip((no_pressing_dist_m - 14.0) / 14.0, 0.0, 1.0)
    s_oob = ball_out_of_bounds.astype(np.float32) * 0.45
    s_struct = out["static_structure_flag"].to_numpy(dtype=np.float32) * 0.35

    raw_score = 0.30 * s_speed + 0.35 * s_walk + 0.15 * s_press + s_oob + s_struct

    if stage1_df is not None:
        # Wzmocnienie o stacjonarność piłki i brak rywalizacji z stage1_tracking.parquet
        stat_norm = np.clip(stage1_df["ball_stationary_frames"].to_numpy(dtype=np.float32) / 35.0, 0.0, 1.0)
        uncontested = 1.0 - stage1_df["is_contested"].to_numpy(dtype=np.float32)
        raw_score = 0.65 * raw_score + 0.25 * stat_norm + 0.10 * uncontested

    # Wygładzenie przyczynowo-symetryczne filtrem kroczącym (1.0s = 25 klatek), aby uniknąć szpilkowych drgań
    kernel = np.ones(25, dtype=np.float32) / 25.0
    smooth_score = np.convolve(np.clip(raw_score, 0.0, 1.0), kernel, mode="same")
    out["kinematic_d_score"] = np.clip(smooth_score, 0.0, 1.0).astype(np.float32)

    return out