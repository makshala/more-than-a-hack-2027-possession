# src/deadball/d_model.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

# Automatyczne dodanie głównego katalogu repozytorium do sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.deadball.kinematic_d import compute_kinematic_deadball_features
from src.deadball.smart_tags import N_FRAMES, build_smart_tag_priors


def _compute_restart_impulse_signal(
    stage1_df: pd.DataFrame,
    restart_hint: np.ndarray,
) -> np.ndarray:
    """
    Oblicza dla każdej klatki siłę impulsu wznowienia gry (moment pierwszego dotknięcia - klatka 0).
    Odporny na brakującą piłkę (ball_status_code == 2): w trakcie długich luk śledzenia
    opiera się na zrywie prędkości gracza wznawiającego (near1_speed_ms) zamiast artefaktów interpolacji.
    """
    ball_speed = stage1_df["ball_speed"].to_numpy(dtype=np.float32)
    ball_acc = stage1_df["ball_acc"].to_numpy(dtype=np.float32)
    ball_stat = stage1_df["ball_stationary_frames"].to_numpy(dtype=np.float32)
    deflection = stage1_df["ball_deflection_angle"].to_numpy(dtype=np.float32)
    status_code = stage1_df["ball_status_code"].to_numpy(dtype=np.int8)

    # Wiarygodność obserwacji piłki: 1.0 gdy OBSERVED (0), 0.7 gdy INTERPOLATED_SHORT (1), 0.15 gdy INFERRED_LATENT (2)
    ball_reliability = np.where(status_code == 0, 1.0, np.where(status_code == 1, 0.70, 0.15)).astype(np.float32)

    # Skok przyspieszenia i prędkości piłki poprzedzony stanem spoczynku (ball_stationary_frames w poprzedniej klatce)
    prev_stat = np.roll(ball_stat, 1)
    prev_stat[0] = 0.0
    was_stationary = np.clip(prev_stat / 10.0, 0.2, 1.0)

    speed_jump = np.maximum(0.0, ball_speed - np.roll(ball_speed, 2))
    speed_jump[:2] = 0.0

    ball_kick_impulse = (
        0.45 * np.clip(ball_acc / 35.0, 0.0, 1.5)
        + 0.35 * np.clip(speed_jump / 8.0, 0.0, 1.5)
        + 0.20 * np.clip(deflection / 1.5, 0.0, 1.0)
    ) * was_stationary * ball_reliability

    # Zryw prędkości zawodnika wykonującego stały fragment gry (H lub A zależnie od restart_team_hint)
    h_dist = stage1_df["home_near1_dist_m"].to_numpy(dtype=np.float32)
    a_dist = stage1_df["away_near1_dist_m"].to_numpy(dtype=np.float32)
    h_spd = stage1_df["home_near1_speed_ms"].to_numpy(dtype=np.float32)
    a_spd = stage1_df["away_near1_speed_ms"].to_numpy(dtype=np.float32)
    h_close = stage1_df["home_near1_closing_speed_ms"].to_numpy(dtype=np.float32)
    a_close = stage1_df["away_near1_closing_speed_ms"].to_numpy(dtype=np.float32)

    # Wybór cech gracza zgodnie z podpowiedzią drużyny wznawiającej (lub bliższego gracza gdy brak)
    use_home = (restart_hint == "H") | ((restart_hint != "A") & (h_dist <= a_dist))
    taker_dist = np.where(use_home, h_dist, a_dist)
    taker_spd = np.where(use_home, h_spd, a_spd)
    taker_close = np.where(use_home, h_close, a_close)

    taker_acc = np.maximum(0.0, taker_spd - np.roll(taker_spd, 3))
    taker_acc[:3] = 0.0

    # Zawodnik jest blisko piłki (< 2.5m) i wykonuje nagły ruch / nabieg
    near_ball_factor = np.clip((3.0 - taker_dist) / 2.0, 0.0, 1.0)
    player_burst_impulse = (
        0.55 * np.clip(taker_acc / 2.5, 0.0, 1.2)
        + 0.45 * np.clip(taker_close / 4.0, 0.0, 1.2)
    ) * near_ball_factor

    # Fuzja impulsu piłki i zawodnika (gdy piłka niewidoczna, ciężar przejmuje kinematyka gracza)
    total_impulse = ball_kick_impulse + (1.0 - 0.6 * ball_reliability) * player_burst_impulse
    return total_impulse.astype(np.float32)


def _snap_restart_frame_zero(
    p_dead_raw: np.ndarray,
    is_sp_window: np.ndarray,
    impulse_signal: np.ndarray,
    kinematic_d_score: np.ndarray,
) -> np.ndarray:
    """
    Dla każdego spójnego okna stałego fragmentu gry (is_set_piece_window) znajduje
    dokładną klatkę zakończenia stanu D ("klatkę 0" wznowienia gry).
    Przed klatką 0 wzmacnia p_dead (trwa przerwa), a od klatki 0 ostro wygasza p_dead
    (piłka w grze), eliminując rozmycie na granicach segmentów.
    """
    p_refined = p_dead_raw.copy()

    # Znalezienie granic spójnych przedziałów is_set_piece_window == True
    padded = np.concatenate([[False], is_sp_window, [False]])
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1]) - 1

    for s_idx, e_idx in zip(starts, ends):
        win_len = e_idx - s_idx + 1
        if win_len < 10:
            continue

        # Szukamy klatki 0 w drugiej połowie lub końcowych 65% okna kotwicy,
        # gdzie wcześniej występował wyraźny stan martwej piłki
        search_start = s_idx + int(0.30 * win_len)
        search_slice = slice(search_start, e_idx + 1)

        local_imp = impulse_signal[search_slice]
        # Klatka 0 powinna następować po fazie niskiej dynamiki (wysokie kinematic_d_score)
        pre_kin = kinematic_d_score[search_slice]
        # Po klatce 0 kinematyka martwej piłki powinna spadać
        post_kin = np.roll(kinematic_d_score, -25)[search_slice]
        kin_drop = np.maximum(0.0, pre_kin - post_kin)

        restart_score = local_imp * (0.5 + 0.5 * pre_kin) + 0.4 * kin_drop

        if restart_score.size > 0 and np.max(restart_score) > 0.22:
            best_offset = int(np.argmax(restart_score))
            frame_zero = search_start + best_offset

            # Wyszukanie początku stanu D wewnątrz okna (moment zatrzymania gry przed wznowieniem)
            pre_slice = kinematic_d_score[s_idx:frame_zero]
            if pre_slice.size > 0:
                high_d_indices = np.flatnonzero(pre_slice > 0.42)
                d_start = s_idx + int(high_d_indices[0]) if high_d_indices.size > 0 else s_idx
            else:
                d_start = s_idx

            # Wzmocnienie p_dead w przedziale [d_start, frame_zero - 1]
            if frame_zero > d_start:
                p_refined[d_start:frame_zero] = np.maximum(
                    p_refined[d_start:frame_zero],
                    np.clip(0.55 + 0.40 * kinematic_d_score[d_start:frame_zero], 0.65, 0.98),
                )

            # Gwałtowne wygaszenie p_dead od klatki 0 (piłka wprowadzona do gry!)
            post_end = min(N_FRAMES, max(e_idx + 1, frame_zero + 50))
            decay_len = post_end - frame_zero
            if decay_len > 0:
                decay_curve = np.linspace(0.05, 0.25, decay_len, dtype=np.float32)
                p_refined[frame_zero:post_end] = np.minimum(
                    p_refined[frame_zero:post_end],
                    decay_curve * kinematic_d_score[frame_zero:post_end],
                )

    return p_refined


def run_deadball_pipeline(
    stage1_parquet_path: Union[str, Path] = "data/interim/stage1_tracking.parquet",
    smart_tags_xml_path: Optional[Union[str, Path]] = None,
    raw_atd_df: Optional[pd.DataFrame] = None,
    output_parquet_path: Union[str, Path] = "data/interim/stage2_deadball.parquet",
) -> pd.DataFrame:
    """
    Główny potok modelu stanu D:
    1. Wczytuje stage1_tracking.parquet oraz priorytety Smart Tagging (XML).
    2. Oblicza wskaźniki kinematyczne braku gry z kinematic_d.py.
    3. Dokonuje fuzji sygnałów i detekcji klatki 0 (pierwszego dotknięcia wznawiającego grę).
    4. Rozwiązuje braki śledzenia piłki i okresy poza grą (period == 0, is_pre_kickoff_d == 1).
    5. Zapisuje wynik do pliku stage2_deadball.parquet.
    """
    stage1_path = Path(stage1_parquet_path)
    if not stage1_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku wejściowego Stage 1: {stage1_path}")

    stage1_df = pd.read_parquet(stage1_path)
    if len(stage1_df) != N_FRAMES:
        raise ValueError(f"Oczekiwano {N_FRAMES} klatek w stage1_tracking.parquet, otrzymano {len(stage1_df)}.")

    # Upewnienie się, że wiersze są posortowane dokładnie według frame_id 0..199318
    if "frame_id" in stage1_df.columns:
        stage1_df = stage1_df.sort_values("frame_id").reset_index(drop=True)

    # 1. Priorytety czasowe ze Smart Tagging (XML + cechy st_* z stage1)
    priors_df = build_smart_tag_priors(
        xml_path=smart_tags_xml_path,
        stage1_df=stage1_df,
    )

    # 2. Cechy kinematyczne z kinematic_d.py
    kin_df = compute_kinematic_deadball_features(
        stage1_df=stage1_df,
        raw_atd_df=raw_atd_df,
    )

    is_sp_window = priors_df["is_set_piece_window"].to_numpy(dtype=bool)
    sp_prior_prob = priors_df["sp_prior_prob"].to_numpy(dtype=np.float32)
    restart_hint = priors_df["restart_team_hint"].to_numpy()
    kin_score = kin_df["kinematic_d_score"].to_numpy(dtype=np.float32)
    ball_oob = kin_df["ball_out_of_bounds"].to_numpy(dtype=bool)

    # 3. Fuzja probabilistyczna (Bayesowska kombinacja kotwicy XML i kinematyki graczy)
    # Wewnątrz okna stałego fragmentu łączymy prior z kinematyką; poza oknem wymagamy
    # silnego dowodu kinematycznego (np. wyjścia piłki poza boisko + zatrzymania graczy)
    ball_status_code = stage1_df["ball_status_code"].to_numpy(dtype=np.int8)
    ball_visible = ball_status_code == 0

    p_in_window = 0.45 * sp_prior_prob + 0.55 * kin_score
    # Poza oknem XML unikamy fałszywych przerw, gdy piłka tylko zniknęła z kamery podczas żywej akcji
    oob_confident = ball_oob & (ball_visible | (kin_score > 0.60))
    p_out_window = np.where(
        oob_confident,
        np.clip(0.50 + 0.45 * kin_score, 0.0, 0.95),
        np.clip((kin_score - 0.35) * 0.85, 0.0, 0.75),
    )

    p_dead_raw = np.where(is_sp_window, p_in_window, p_out_window).astype(np.float32)

    # Jeżeli bramkarz trzyma piłkę w rękach w grze (gk_holding_ball_proxy != 0), piłka JEST w grze (żywa)
    if "gk_holding_ball_proxy" in stage1_df.columns:
        gk_hold = stage1_df["gk_holding_ball_proxy"].to_numpy(dtype=np.int8) != 0
        p_dead_raw = np.where(gk_hold & ~is_sp_window, p_dead_raw * 0.25, p_dead_raw)

    # 4. Detekcja klatki 0 (pierwszego kontaktu wznawiającego grę) i wyostrzenie granic stanu D
    impulse_signal = _compute_restart_impulse_signal(stage1_df, restart_hint)
    p_dead = _snap_restart_frame_zero(
        p_dead_raw=p_dead_raw,
        is_sp_window=is_sp_window,
        impulse_signal=impulse_signal,
        kinematic_d_score=kin_score,
    )

    # 5. Twarde reguły domenowe dla czasu poza grą (period == 0, is_pre_kickoff_d, st_post_goal_dead)
    if "period" in stage1_df.columns:
        non_play_period = stage1_df["period"].to_numpy(dtype=np.int8) == 0
        p_dead = np.where(non_play_period, 1.0, p_dead)

    if "is_pre_kickoff_d" in stage1_df.columns:
        pre_ko = stage1_df["is_pre_kickoff_d"].to_numpy(dtype=np.int8) == 1
        p_dead = np.where(pre_ko, 1.0, p_dead)

    if "st_post_goal_dead" in stage1_df.columns:
        post_goal = stage1_df["st_post_goal_dead"].to_numpy(dtype=np.int8) == 1
        p_dead = np.where(post_goal, 0.99, p_dead)

    # Zabezpieczenie przed jakimikolwiek wartościami NaN / Inf oraz obcięcie do [0.0, 1.0]
    p_dead = np.nan_to_num(p_dead, nan=0.0, posinf=1.0, neginf=0.0)
    p_dead = np.clip(p_dead, 0.0, 1.0).astype(np.float32)

    # Uzupełnienie restart_team_hint dla wykrytych przerw spoza XML na podstawie przewagi odległości do piłki
    missing_hint_mask = pd.isna(restart_hint) & (p_dead >= 0.50)
    if np.any(missing_hint_mask) and "near1_dist_diff_m" in stage1_df.columns:
        # Dodatnie near1_dist_diff_m oznacza, że gracz Gospodarzy (H) jest bliżej piłki
        dist_diff = stage1_df["near1_dist_diff_m"].to_numpy(dtype=np.float32)
        restart_hint = restart_hint.astype(object)
        restart_hint[missing_hint_mask & (dist_diff >= 0.0)] = "H"
        restart_hint[missing_hint_mask & (dist_diff < 0.0)] = "A"

    # Konstrukcja finalnego DataFrame zgodnego ze specyfikacją wyjściową
    result_df = pd.DataFrame({
        "frame": np.arange(N_FRAMES, dtype=np.int32),
        "p_dead": p_dead,
        "is_set_piece_window": is_sp_window.astype(bool),
        "restart_team_hint": pd.Categorical(restart_hint, categories=["H", "A"]),
    })

    # Walidacja kontraktu wyjściowego przed zapisem
    assert len(result_df) == N_FRAMES, f"Błędna liczba wierszy: {len(result_df)} != {N_FRAMES}"
    assert not result_df["frame"].isna().any(), "Kolumna 'frame' zawiera wartości NaN!"
    assert not result_df["p_dead"].isna().any(), "Kolumna 'p_dead' zawiera wartości NaN!"
    assert not result_df["is_set_piece_window"].isna().any(), "Kolumna 'is_set_piece_window' zawiera NaN!"
    assert (result_df["frame"].to_numpy() == np.arange(N_FRAMES)).all(), "Naruszenie kolejności klatek 0..199318!"

    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)

    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generowanie stage2_deadball.parquet (detekcja stanu D).")
    parser.add_argument(
        "--stage1",
        type=str,
        default="data/interim/stage1_tracking.parquet",
        help="Ścieżka do pliku stage1_tracking.parquet",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default="data/tactical/BAR-ATH_tactical_smart_tagging.xml",
        help="Opcjonalna ścieżka do pliku BAR-ATH_tactical_smart_tagging.xml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/interim/stage2_deadball.parquet",
        help="Ścieżka wyjściowa dla pliku stage2_deadball.parquet",
    )
    args = parser.parse_args()

    xml_file = args.xml if Path(args.xml).exists() else None
    df_out = run_deadball_pipeline(
        stage1_parquet_path=args.stage1,
        smart_tags_xml_path=xml_file,
        output_parquet_path=args.output,
    )
    print(f"Zapisano pomyślnie {len(df_out)} klatek do {args.output}.")
    print(df_out.info())