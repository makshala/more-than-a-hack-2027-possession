"""
Production Possession Decoding Pipeline & Tactical Analytics Engine.
Fixes spell under-segmentation (over-smoothing) and D-boundary recall loss:
1. Recovers valid Dead-Ball frames where p_dead is the modal class or near set-piece edges.
2. Uses dual-scale Savitzky/Boxcar probability smoothing + calibrated Log-Viterbi trellis.
3. Never bridges across genuine Dead-Ball stoppages (preserves H -> D -> H as two possessions).
4. Bypasses the destructive 1600ms unconditional sandwich-merge bug in spells.py.
"""
from typing import Dict, Any, List, Tuple, Optional
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from hsmm_decoder import HSMMExplicitDurationDecoder
from config import (
    DEFAULT_FPS, STATE_HOME, STATE_AWAY, STATE_DEAD_BALL, STATE_MAP,
    DEFAULT_PARQUET_PATH, DEFAULT_SYNC_PATH,
    DEFAULT_OUTPUT_SPELLS_PARQUET, DEFAULT_OUTPUT_SPELLS_CSV,
    normalize_probabilities_df
)
from spells import PossessionSpellExtractor
from diagnostics import TemporalDiagnostics

# Official Competition Frame Constants (DATA.md & FAQ.md)
TOTAL_MATCH_FRAMES: int = 199_319
H1_START: int = 11_185
H1_END: int = 90_651
H2_START: int = 110_808
H2_KICKOFF: int = 113_742
H2_END: int = 190_197
DEFAULT_SUBMISSION_CSV: str = "evaluation/results/IAD AGH.csv"


def _resolve_match_halves(sync_path: Optional[str]) -> Dict[str, Dict[str, int]]:
    default_bounds = {
        "half_1": {"start_frame": H1_START, "end_frame": H1_END},
        "half_2": {"start_frame": H2_START, "end_frame": H2_END},
    }
    if sync_path and os.path.exists(sync_path):
        try:
            with open(sync_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if (
                isinstance(data, dict)
                and "half_1" in data
                and "half_2" in data
                and "start_frame" in data["half_1"]
                and "start_frame" in data["half_2"]
            ):
                return data
        except Exception:
            pass
    return default_bounds


def _smooth_1d_reflect(arr: np.ndarray, window: int) -> np.ndarray:
    """Zero-phase symmetric moving average with reflection padding (zero boundary lag)."""
    if window <= 1 or len(arr) == 0:
        return arr.astype(np.float64, copy=True)
    pad = window // 2
    padded = np.pad(arr.astype(np.float64), (pad, pad), mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def _clean_segments_spell_aware(
    states: np.ndarray,
    probs: np.ndarray,
    min_live_frames: int = 25,
    min_dead_frames: int = 25,
    live_sandwich_frames: int = 55,
) -> np.ndarray:
    """
    SpellF1-aware segment post-processor:
    - Protects the 14s mean duration by sweeping out transient (<2.2s) H->A->H flickers.
    - Prevents false boundary cuts created by momentarily high p_dead that don't form a real stoppage.
    """
    n = len(states)
    if n <= 1:
        return states.copy()

    def _to_rle(arr: np.ndarray) -> List[List[int]]:
        diffs = np.flatnonzero(arr[1:] != arr[:-1]) + 1
        starts = np.concatenate(([0], diffs))
        ends = np.concatenate((diffs - 1, [len(arr) - 1]))
        return [[int(arr[s]), int(s), int(e)] for s, e in zip(starts, ends)]

    def _consolidate(segs: List[List[int]]) -> List[List[int]]:
        out: List[List[int]] = []
        for st, s, e in segs:
            if out and out[-1][0] == st:
                out[-1][2] = e
            else:
                out.append([st, s, e])
        return out

    segs = _to_rle(states)

    # Pass 1: Prune tiny, unconfirmed D states (< 1.0s) that clip possession lengths.
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            st, s, e = segs[i]
            dur = e - s + 1
            if st == STATE_DEAD_BALL and dur < min_dead_frames and len(segs) > 1:
                mean_h = float(np.mean(probs[s : e + 1, STATE_HOME]))
                mean_a = float(np.mean(probs[s : e + 1, STATE_AWAY]))
                segs[i][0] = STATE_HOME if mean_h >= mean_a else STATE_AWAY
                changed = True
                break
        if changed:
            segs = _consolidate(segs)

    # Pass 2: Aggressive sandwich bridge (H->A->H) up to 2.2s (55 frames) to repair fragmented turnovers.
    changed = True
    while changed:
        changed = False
        for i in range(1, len(segs) - 1):
            st, s, e = segs[i]
            dur = e - s + 1
            prev_st = segs[i - 1][0]
            next_st = segs[i + 1][0]
            if (
                st in (STATE_HOME, STATE_AWAY)
                and prev_st == next_st
                and prev_st in (STATE_HOME, STATE_AWAY)
                and prev_st != st
                and dur <= live_sandwich_frames
            ):
                segs[i][0] = prev_st
                changed = True
                break
        if changed:
            segs = _consolidate(segs)

    # Pass 3: Micro-possession absorption (< 1.0s)
    changed = True
    while changed:
        changed = False
        if len(segs) <= 1:
            break
        for i in range(len(segs)):
            st, s, e = segs[i]
            dur = e - s + 1
            if st in (STATE_HOME, STATE_AWAY) and dur < min_live_frames:
                if 0 < i < len(segs) - 1:
                    left_st = segs[i - 1][0]
                    right_st = segs[i + 1][0]
                    p_left = float(np.mean(probs[s : e + 1, left_st]))
                    p_right = float(np.mean(probs[s : e + 1, right_st]))
                    segs[i][0] = left_st if p_left >= p_right else right_st
                elif i > 0:
                    segs[i][0] = segs[i - 1][0]
                else:
                    segs[i][0] = segs[i + 1][0]
                changed = True
                break
        if changed:
            segs = _consolidate(segs)

    out_states = np.empty(n, dtype=np.int32)
    for st, s, e in segs:
        out_states[s : e + 1] = st
    return out_states

def _dilate_live_play(states: np.ndarray, expand_frames: int = 12) -> np.ndarray:
    
    n = len(states)
    if n == 0:
        return states

    diffs = np.flatnonzero(states[1:] != states[:-1]) + 1
    starts = np.concatenate(([0], diffs))
    ends = np.concatenate((diffs - 1, [n - 1]))
    
    out = states.copy()
    
    for s, e in zip(starts, ends):
        st = states[s]
        if st in (0, 1):  # H (0) lub A (1)
            # Ekspansja w lewo (nadpisuje tylko D)
            for i in range(s - 1, max(-1, s - expand_frames - 1), -1):
                if out[i] == 2:  # 2 to stan D
                    out[i] = st
                else:
                    break
                    
            # Ekspansja w prawo (nadpisuje tylko D)
            for i in range(e + 1, min(n, e + expand_frames + 1)):
                if out[i] == 2:
                    out[i] = st
                else:
                    break
    return out

def _decode_half_spell_optimal(df_half: pd.DataFrame) -> np.ndarray:
    p_h = df_half["p_home"].to_numpy(dtype=np.float64)
    p_a = df_half["p_away"].to_numpy(dtype=np.float64)
    dead_col = "p_dead" if "p_dead" in df_half.columns else "p_dead_ball"
    p_d = df_half[dead_col].to_numpy(dtype=np.float64)

    # 1. Algorytmiczny Time-Shifting (skalowanie 10 -> 25 FPS)
    shift = 17
    if shift > 0:
        p_h = np.pad(p_h[shift:], (0, shift), mode='edge')
        p_a = np.pad(p_a[shift:], (0, shift), mode='edge')
        p_d = np.pad(p_d[shift:], (0, shift), mode='edge')

    # 2. Standardowe algorytmiczne wygładzanie sygnałów
    p_h_smooth = _smooth_1d_reflect(p_h, window=5)
    p_a_smooth = _smooth_1d_reflect(p_a, window=5)
    p_d_smooth = _smooth_1d_reflect(p_d, window=4)

    probs = np.stack([p_h_smooth, p_a_smooth, p_d_smooth], axis=1)
    probs = np.clip(probs, 0.02, 0.98)
    probs /= np.sum(probs, axis=1, keepdims=True)
    log_emit = np.log(probs)

    n = len(df_half)
    
    # 3. Kary przejść baselinowo wyliczone przez Optunę, potem dostosowywane do wynikow na
    # syntetycznym GT
    pen_ha = 1.6
    pen_hd = 2.5
    
    log_trans = np.array([
        [0.0, -pen_ha, -pen_hd],
        [-pen_ha, 0.0, -pen_hd],
        [-pen_hd, -pen_hd, 0.0],
    ], dtype=np.float64)

    dp = np.zeros((n, 3), dtype=np.float64)
    back = np.zeros((n, 3), dtype=np.int32)
    dp[0] = log_emit[0]

    for t in range(1, n):
        scores = dp[t - 1][:, None] + log_trans
        back[t] = np.argmax(scores, axis=0)
        dp[t] = np.max(scores, axis=0) + log_emit[t]

    states = np.zeros(n, dtype=np.int32)
    states[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]

    # 4. Standardowe algorytmiczne czyszczenie segmentów
    states_clean = _clean_segments_spell_aware(
        states=states,
        probs=probs,
        min_live_frames=18,         
        min_dead_frames=25,         
        live_sandwich_frames=27,    
    )

    states_clean = _dilate_live_play(states_clean, expand_frames=5)
    
    return states_clean
    
def _decode_half_hsmm(df_half: pd.DataFrame, fps: float = DEFAULT_FPS) -> np.ndarray:
    """
    Dekoduje połowę meczu za pomocą jawnego dekodera czasowego HSMM (Gamma Duration Modeling).
    """
    # Bazowa macierz przejść uwzględniająca zwroty akcji przez martwą piłkę
    A = np.array([
        [0.96, 0.01, 0.03],
        [0.01, 0.96, 0.03],
        [0.05, 0.05, 0.90]
    ], dtype=np.float64)

    decoder = HSMMExplicitDurationDecoder(
        transition_matrix=A,
        duration_weight=1.2,          # Waga kary za zgodność z rozkładem czasu trwania
        input_semantics="posterior",  # Dokładna inwersja bayesowska z Roli 4
        fps=fps
    )
    return decoder.decode(df_half)

def _build_spells_dataframe_direct(
    df_source: pd.DataFrame,
    clean_states: np.ndarray,
    phase_name: str,
    fps: float = DEFAULT_FPS,
) -> pd.DataFrame:
    """
    Constructs the spells DataFrame directly from clean_states WITHOUT calling
    PossessionSpellExtractor.extract_spells() (which contained a hidden bug that
    unconditionally re-ran 1600ms sandwich merging over D states).
    """
    if len(clean_states) == 0:
        return pd.DataFrame()

    diffs = np.flatnonzero(clean_states[1:] != clean_states[:-1]) + 1
    start_indices = np.concatenate(([0], diffs))
    end_indices = np.concatenate((diffs - 1, [len(clean_states) - 1]))

    frame_ids = df_source["frame_id"].to_numpy(dtype=np.int64)
    timestamps = (
        df_source["timestamp"].to_numpy(dtype=np.float64)
        if "timestamp" in df_source.columns
        else np.around(frame_ids / fps, decimals=4)
    )
    dead_col = "p_dead" if "p_dead" in df_source.columns else "p_dead_ball"
    probs = df_source[["p_home", "p_away", dead_col]].to_numpy(dtype=np.float64)

    records: List[Dict[str, Any]] = []
    for spell_id, (s_idx, e_idx) in enumerate(zip(start_indices, end_indices)):
        st_int = int(clean_states[s_idx])
        start_f = int(frame_ids[s_idx])
        end_f = int(frame_ids[e_idx])
        start_ts = float(timestamps[s_idx])
        end_ts = float(timestamps[e_idx])
        dur_frames = int(e_idx - s_idx + 1)
        dur_s = float(np.around(dur_frames / fps, decimals=4))

        sub_p = probs[s_idx : e_idx + 1]
        records.append({
            "spell_id": spell_id,
            "state": STATE_MAP.get(st_int, "D"),
            "half": phase_name,
            "start_frame": start_f,
            "end_frame": end_f,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "duration_frames": dur_frames,
            "duration_seconds": dur_s,
            "duration_ms": float(np.around(dur_s * 1000.0, decimals=1)),
            "mean_confidence": float(np.around(np.mean(sub_p[:, st_int]), decimals=4)),
            "p_home_mean": float(np.around(np.mean(sub_p[:, 0]), decimals=4)),
            "p_away_mean": float(np.around(np.mean(sub_p[:, 1]), decimals=4)),
            "p_dead_mean": float(np.around(np.mean(sub_p[:, 2]), decimals=4)),
        })
    return pd.DataFrame(records)


def build_and_validate_submission_csv(
    df_spells: pd.DataFrame,
    sync_metadata: Dict[str, Dict[str, int]],
    out_submission_csv: str,
    total_frames: int = TOTAL_MATCH_FRAMES,
) -> pd.DataFrame:
    labels = np.full(total_frames, "X", dtype="<U1")

    for row in df_spells.itertuples(index=False):
        s_f = max(0, int(row.start_frame))
        e_f = min(total_frames - 1, int(row.end_frame))
        st_str = str(row.state).strip().upper()
        if st_str in ("H", "A", "D") and e_f >= s_f:
            labels[s_f : e_f + 1] = st_str

    h1_s = int(sync_metadata["half_1"]["start_frame"])
    h1_e = int(sync_metadata["half_1"]["end_frame"])
    h2_s = int(sync_metadata["half_2"]["start_frame"])
    h2_e = int(sync_metadata["half_2"]["end_frame"])

    graded_mask = np.zeros(total_frames, dtype=bool)
    graded_mask[h1_s : h1_e + 1] = True
    graded_mask[h2_s : h2_e + 1] = True
    labels[~graded_mask] = "X"

    if h2_s <= H2_KICKOFF - 1 <= h2_e:
        labels[h2_s:H2_KICKOFF] = "D"

    for half_s, half_e in ((h1_s, h1_e), (h2_s, h2_e)):
        half_slice = labels[half_s : half_e + 1]
        if np.any(half_slice == "X"):
            half_slice[half_slice == "X"] = "D"
            labels[half_s : half_e + 1] = half_slice

    df_sub = pd.DataFrame({
        "frame": np.arange(total_frames, dtype=np.int32),
        "label": labels,
    })

    assert list(df_sub.columns) == ["frame", "label"]
    assert len(df_sub) == total_frames
    assert (df_sub["frame"].to_numpy() == np.arange(total_frames)).all()
    assert not df_sub["label"].isna().any()
    assert set(df_sub["label"].unique()).issubset({"H", "A", "D", "X"})

    out_dir = os.path.dirname(os.path.abspath(out_submission_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_sub.to_csv(out_submission_csv, index=False, encoding="utf-8", lineterminator="\n")
    print(f"[Submission Exporter] Official submission CSV written to: {out_submission_csv} ({len(df_sub):,} rows)")
    return df_sub


def run_real_match_pipeline(
    parquet_path: str = DEFAULT_PARQUET_PATH,
    sync_path: Optional[str] = DEFAULT_SYNC_PATH,
    out_parquet: Optional[str] = DEFAULT_OUTPUT_SPELLS_PARQUET,
    out_csv: Optional[str] = DEFAULT_OUTPUT_SPELLS_CSV,
    out_submission_csv: Optional[str] = DEFAULT_SUBMISSION_CSV,
    fps: float = DEFAULT_FPS,
    use_hsmm: bool = False
) -> pd.DataFrame:
    print("=" * 80)
    print(" ROLE 5: SPELL-OPTIMAL VITERBI POSSESSION DECODING (PRODUCTION PIPELINE)")
    print("=" * 80)

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Role 4 telemetry file not found: {parquet_path}")

    print(f"\n[1/5] Loading Role 4 Telemetry: {parquet_path}")
    t0 = time.time()
    df_raw = pd.read_parquet(parquet_path)
    df_match = normalize_probabilities_df(df_raw, fps=fps)
    n_frames = len(df_match)
    print(f"      Loaded {n_frames:,} frames in {time.time()-t0:.2f}s")

    print(f"\n[2/5] Resolving Match Half Boundaries")
    sync_metadata = _resolve_match_halves(sync_path)
    h1_s = sync_metadata["half_1"]["start_frame"]
    h1_e = sync_metadata["half_1"]["end_frame"]
    h2_s = sync_metadata["half_2"]["start_frame"]
    h2_e = sync_metadata["half_2"]["end_frame"]
    h2_play_start = max(h2_s, min(H2_KICKOFF, h2_e))

    print(f"\n[3/5] Executing Spell-Optimal Multi-Scale Viterbi Decoder...")
    t_decode_start = time.time()
    decoded_states = np.full(n_frames, STATE_DEAD_BALL, dtype=np.int32)

    df_h1 = df_match.iloc[h1_s : h1_e + 1].copy().reset_index(drop=True)
    decoded_states[h1_s : h1_e + 1] = _decode_half_spell_optimal(df_h1)

    df_h2 = df_match.iloc[h2_play_start : h2_e + 1].copy().reset_index(drop=True)
    decoded_states[h2_play_start : h2_e + 1] = _decode_half_spell_optimal(df_h2)

    decoded_states[h2_s:h2_play_start] = STATE_DEAD_BALL
    print(f"      Decoding completed in {time.time() - t_decode_start:.2f}s")

    print("\n[4/5] Building Discrete Tactical Spells Table...")
    df_spells_h1 = _build_spells_dataframe_direct(
        df_source=df_match.iloc[h1_s : h1_e + 1].reset_index(drop=True),
        clean_states=decoded_states[h1_s : h1_e + 1],
        phase_name="half_1",
        fps=fps,
    )
    df_spells_h2_pre = _build_spells_dataframe_direct(
        df_source=df_match.iloc[h2_s:h2_play_start].reset_index(drop=True),
        clean_states=decoded_states[h2_s:h2_play_start],
        phase_name="half_2",
        fps=fps,
    )
    df_spells_h2_play = _build_spells_dataframe_direct(
        df_source=df_match.iloc[h2_play_start : h2_e + 1].reset_index(drop=True),
        clean_states=decoded_states[h2_play_start : h2_e + 1],
        phase_name="half_2",
        fps=fps,
    )

    df_spells = pd.concat([df_spells_h1, df_spells_h2_pre, df_spells_h2_play], ignore_index=True)
    df_spells["spell_id"] = np.arange(len(df_spells), dtype=np.int32)
    print(f"      Extracted {len(df_spells):,} discrete tactical spells")

    print("\n[5/5] Exporting Spells & Official Competition Submission CSV...")
    PossessionSpellExtractor.export_spells(df_spells, parquet_path=out_parquet, csv_path=out_csv)

    if out_submission_csv:
        build_and_validate_submission_csv(
            df_spells=df_spells,
            sync_metadata=sync_metadata,
            out_submission_csv=out_submission_csv,
            total_frames=n_frames,
        )

    return df_spells


def main():
    parser = argparse.ArgumentParser(description="Role 5: Tactical Possession Temporal Decoder & Submission Builder")
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--sync", type=str, default=DEFAULT_SYNC_PATH)
    parser.add_argument("--out-parquet", type=str, default=DEFAULT_OUTPUT_SPELLS_PARQUET)
    parser.add_argument("--out-csv", type=str, default=DEFAULT_OUTPUT_SPELLS_CSV)
    parser.add_argument("--out-submission", type=str, default=DEFAULT_SUBMISSION_CSV)
    parser.add_argument("--viterbi-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()
    run_real_match_pipeline(
        parquet_path=args.parquet,
        sync_path=args.sync,
        out_parquet=args.out_parquet,
        out_csv=args.out_csv,
        out_submission_csv=args.out_submission,
    )


if __name__ == "__main__":
    main()