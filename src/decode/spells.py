"""
Possession Spell Extraction with 1D Morphological Filtering and Temporal Hysteresis.
Applies temporal closing and opening to bridge tracking micro-gaps and remove transient flickers,
snapping transition boundaries to the nearest stable tracking frames to resolve borderline uncertainty.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_opening
from config import (
    DEFAULT_FPS, STATE_MAP, INV_STATE_MAP, ms_to_frames, frames_to_ms,
    DEFAULT_CLOSING_KERNEL_MS, DEFAULT_OPENING_KERNEL_MS,
    HYSTERESIS_CERTAINTY_THRESHOLD, HYSTERESIS_MARGIN_THRESHOLD
)


class PossessionSpellExtractor:
    """
    Tactical possession spell extractor implementing:
    1. 1D Temporal Morphological Filtering (Closing + Opening).
    2. Temporal Hysteresis & Stable-Frame Edge Snapping.
    3. Run-length boundary resolution into continuous tactical blocks.
    """

    @staticmethod
    def _morphological_filter_1d(
        binary_mask: np.ndarray,
        closing_frames: int,
        opening_frames: int
    ) -> np.ndarray:
        """
        Applies 1D mathematical morphology with boundary preservation:
        - Closing: Dilation followed by Erosion (bridges gaps <= closing_frames)
        - Opening: Erosion followed by Dilation (erodes spikes <= opening_frames)
        """
        mask = np.asarray(binary_mask, dtype=bool)
        n = len(mask)
        if n == 0:
            return mask

        res = mask.copy()
        # 1. Closing: bridges small zero-gaps inside a sustained possession block
        if closing_frames > 1:
            border = int(res[0])
            res = binary_closing(res, structure=np.ones(closing_frames, dtype=bool), border_value=border)

        # 2. Opening: removes isolated transient spikes shorter than opening_frames
        if opening_frames > 1:
            res = binary_opening(res, structure=np.ones(opening_frames, dtype=bool))

        return res

    @classmethod
    def apply_morphological_filtering(
        cls,
        decoded_states: np.ndarray,
        df_probs: Optional[pd.DataFrame] = None,
        closing_ms: float = DEFAULT_CLOSING_KERNEL_MS,
        opening_ms: float = DEFAULT_OPENING_KERNEL_MS,
        fps: float = DEFAULT_FPS
    ) -> np.ndarray:
        """
        Applies multi-class 1D morphological closing and opening across possession states.
        Resolves small tracking gaps and momentary flickers into contiguous blocks.
        """
        n = len(decoded_states)
        if n <= 1:
            return decoded_states.copy()

        closing_frames = ms_to_frames(closing_ms, fps)
        opening_frames = ms_to_frames(opening_ms, fps)

        filtered_states = decoded_states.copy()
        masks = np.zeros((3, n), dtype=bool)

        for s in range(3):
            binary_s = (decoded_states == s)
            masks[s] = cls._morphological_filter_1d(binary_s, closing_frames, opening_frames)

        # Resolve conflicts and unassigned frames
        # If df_probs is provided, use smoothed posterior probabilities as arbiter
        dead_col = None
        if df_probs is not None:
            if "p_dead_ball" in df_probs.columns:
                dead_col = "p_dead_ball"
            elif "p_dead" in df_probs.columns:
                dead_col = "p_dead"

        if df_probs is not None and dead_col and all(c in df_probs.columns for c in ["p_home", "p_away", dead_col]):
            probs = df_probs[["p_home", "p_away", dead_col]].to_numpy(dtype=np.float64)
        else:
            probs = np.zeros((n, 3), dtype=np.float64)
            for s in range(3):
                probs[:, s] = (decoded_states == s).astype(np.float64)

        for t in range(n):
            active_states = np.where(masks[:, t])[0]
            if len(active_states) == 1:
                filtered_states[t] = active_states[0]
            elif len(active_states) > 1:
                # Multiple active: choose highest probability
                filtered_states[t] = active_states[np.argmax(probs[t, active_states])]
            else:
                # No active state after erosion: fallback to previous state or highest prob
                if t > 0:
                    filtered_states[t] = filtered_states[t - 1]
                else:
                    filtered_states[t] = np.argmax(probs[t])

        return filtered_states

    @classmethod
    def apply_temporal_hysteresis_snapping(
        cls,
        decoded_states: np.ndarray,
        df_probs: pd.DataFrame,
        window_ms: float = 600.0,
        fps: float = DEFAULT_FPS,
        certainty_thresh: float = HYSTERESIS_CERTAINTY_THRESHOLD
    ) -> np.ndarray:
        """
        Snaps decoded state transition boundaries to the nearest stable tracking frame
        where upstream certainty is high and probability gradient is maximal.
        """
        n = len(decoded_states)
        if n <= 2:
            return decoded_states.copy()

        dead_col = "p_dead_ball" if "p_dead_ball" in df_probs.columns else "p_dead" if "p_dead" in df_probs.columns else None
        required_cols = ["p_home", "p_away", dead_col] if dead_col else []
        if not dead_col or not all(c in df_probs.columns for c in required_cols):
            return decoded_states.copy()

        posteriors = df_probs[["p_home", "p_away", dead_col]].to_numpy(dtype=np.float64)
        max_certainty = np.max(posteriors, axis=1)

        window_frames = ms_to_frames(window_ms, fps)
        snapped_states = decoded_states.copy()

        # Find transition boundary points
        boundaries = np.where(np.diff(snapped_states) != 0)[0] + 1

        for b_idx in boundaries:
            prev_s = snapped_states[b_idx - 1]
            next_s = snapped_states[b_idx]

            # Search in local temporal window [b_idx - window_frames, b_idx + window_frames]
            w_start = max(0, b_idx - window_frames)
            w_end = min(n, b_idx + window_frames + 1)

            if w_end - w_start <= 1:
                continue

            # Compute probability margin: P(next_s) - P(prev_s)
            margin = posteriors[w_start:w_end, next_s] - posteriors[w_start:w_end, prev_s]
            
            # The ideal transition occurs at zero-crossing of margin or maximum gradient
            # Weighted by stability / certainty of the tracking signal
            stability_weight = max_certainty[w_start:w_end]
            
            # Find the transition inflection point
            # Discrete gradient of margin
            grad = np.gradient(margin)
            gradient_scores = np.abs(grad) * stability_weight
            
            # Candidate points where margin crosses zero (sign change)
            zero_crossings = np.where(margin[:-1] * margin[1:] <= 0)[0]
            if len(zero_crossings) > 0:
                best_local_offset = zero_crossings[np.argmax(stability_weight[zero_crossings])] + 1
            else:
                best_local_offset = np.argmax(gradient_scores)

            ideal_boundary = w_start + best_local_offset
            # Snap boundary
            if ideal_boundary < b_idx:
                snapped_states[ideal_boundary:b_idx] = next_s
            elif ideal_boundary > b_idx:
                snapped_states[b_idx:ideal_boundary] = prev_s

        return snapped_states

    @classmethod
    def extract_spells(
        cls,
        df_source: pd.DataFrame,
        decoded_states: np.ndarray,
        apply_morphology: bool = True,
        apply_hysteresis: bool = True,
        closing_ms: float = DEFAULT_CLOSING_KERNEL_MS,
        opening_ms: float = DEFAULT_OPENING_KERNEL_MS,
        sync_metadata: Optional[Dict[str, Any]] = None,
        fps: float = DEFAULT_FPS
    ) -> pd.DataFrame:
        """
        Full spell extraction pipeline:
        1. 1D Morphological filtering (temporal closing + opening).
        2. Temporal hysteresis & stable-frame edge snapping.
        3. Continuous block aggregation into structured tactical spells DataFrame.
        4. Match phase tagging (half_1, half_2, half_time, pre_match, post_match).
        """
        if len(df_source) != len(decoded_states):
            raise ValueError("Dataframe length and states length must match.")

        if len(decoded_states) == 0:
            return pd.DataFrame(columns=[
                "spell_id", "state", "half", "start_frame", "end_frame",
                "start_timestamp", "end_timestamp", "duration_frames",
                "duration_seconds", "duration_ms", "mean_confidence",
                "p_home_mean", "p_away_mean", "p_dead_mean"
            ])

        clean_states = decoded_states.copy()

        # Step 1: 1D Morphological Filtering
        if apply_morphology:
            clean_states = cls.apply_morphological_filtering(
                clean_states, df_probs=df_source, closing_ms=closing_ms,
                opening_ms=opening_ms, fps=fps
            )

        # Step 2: Temporal Hysteresis & Boundary Edge Snapping
        dead_col = "p_dead_ball" if "p_dead_ball" in df_source.columns else "p_dead" if "p_dead" in df_source.columns else None
        if apply_hysteresis and dead_col and all(c in df_source.columns for c in ["p_home", "p_away", dead_col]):
            clean_states = cls.apply_temporal_hysteresis_snapping(
                clean_states, df_probs=df_source, window_ms=closing_ms, fps=fps
            )

        # Step 3: Run-Length Aggregation into Continuous Tactical Spells
        state_changes = np.diff(clean_states) != 0
        split_indices = np.where(state_changes)[0] + 1

        start_indices = np.insert(split_indices, 0, 0)
        end_indices = np.append(split_indices - 1, len(clean_states) - 1)

        # Frame IDs & Timestamps
        if "frame_id" in df_source.columns:
            frame_ids = df_source["frame_id"].to_numpy()
        else:
            frame_ids = np.arange(len(df_source), dtype=np.int64)

        if "timestamp" in df_source.columns:
            timestamps = df_source["timestamp"].to_numpy()
        else:
            timestamps = np.around(frame_ids / fps, decimals=4)

        has_probs = dead_col and all(c in df_source.columns for c in ["p_home", "p_away", dead_col])
        if has_probs:
            probs = df_source[["p_home", "p_away", dead_col]].to_numpy()
        else:
            probs = None

        # Parse sync_metadata for half classification
        h1_start, h1_end = None, None
        h2_start, h2_end = None, None
        if sync_metadata:
            if "half_1" in sync_metadata:
                h1_start = sync_metadata["half_1"].get("start_frame")
                h1_end = sync_metadata["half_1"].get("end_frame")
            if "half_2" in sync_metadata:
                h2_start = sync_metadata["half_2"].get("start_frame")
                h2_end = sync_metadata["half_2"].get("end_frame")

        spells: List[Dict] = []
        for spell_id, (s_idx, e_idx) in enumerate(zip(start_indices, end_indices)):
            state_int = int(clean_states[s_idx])
            start_f = int(frame_ids[s_idx])
            end_f = int(frame_ids[e_idx])
            start_ts = float(timestamps[s_idx])
            end_ts = float(timestamps[e_idx])
            duration_frames = int(e_idx - s_idx + 1)
            duration_s = np.around(end_ts - start_ts + (1.0 / fps), decimals=4)
            duration_ms = float(np.around(duration_s * 1000.0, decimals=1))

            mean_conf = 1.0
            p_h_mean, p_a_mean, p_d_mean = 0.0, 0.0, 0.0
            if probs is not None:
                mean_conf = float(np.mean(probs[s_idx:e_idx + 1, state_int]))
                p_h_mean = float(np.mean(probs[s_idx:e_idx + 1, 0]))
                p_a_mean = float(np.mean(probs[s_idx:e_idx + 1, 1]))
                p_d_mean = float(np.mean(probs[s_idx:e_idx + 1, 2]))

            # Half / Match Phase tagging
            phase = "match"
            if h1_start is not None and h1_end is not None and h2_start is not None and h2_end is not None:
                mid_f = (start_f + end_f) // 2
                if mid_f < h1_start:
                    phase = "pre_match"
                elif h1_start <= mid_f <= h1_end:
                    phase = "half_1"
                elif h1_end < mid_f < h2_start:
                    phase = "half_time"
                elif h2_start <= mid_f <= h2_end:
                    phase = "half_2"
                else:
                    phase = "post_match"

            spells.append({
                "spell_id": spell_id,
                "state": STATE_MAP.get(state_int, "UNKNOWN"),
                "half": phase,
                "start_frame": start_f,
                "end_frame": end_f,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "duration_frames": duration_frames,
                "duration_seconds": duration_s,
                "duration_ms": duration_ms,
                "mean_confidence": np.around(mean_conf, decimals=4),
                "p_home_mean": np.around(p_h_mean, decimals=4),
                "p_away_mean": np.around(p_a_mean, decimals=4),
                "p_dead_mean": np.around(p_d_mean, decimals=4)
            })

        return pd.DataFrame(spells)

    @staticmethod
    def export_spells(
        df_spells: pd.DataFrame,
        parquet_path: Optional[str] = None,
        csv_path: Optional[str] = None
    ) -> None:
        """Exports extracted spells to Parquet and CSV formats for downstream integration."""
        if parquet_path:
            df_spells.to_parquet(parquet_path, index=False)
            print(f"[Exporter] Successfully exported {len(df_spells)} spells to Parquet: {parquet_path}")
        if csv_path:
            df_spells.to_csv(csv_path, index=False)
            print(f"[Exporter] Successfully exported {len(df_spells)} spells to CSV: {csv_path}")
