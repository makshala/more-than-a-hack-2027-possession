"""
Spell extraction and boundary calculation (RLE).
Addresses task 7: Spells counting, length, and boundaries.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from config import STATE_MAP


class PossessionSpellExtractor:
    """Converts discrete frame states into continuous tactical possession spells."""

    @staticmethod
    def extract_spells(df_source: pd.DataFrame, decoded_states: np.ndarray) -> pd.DataFrame:
        if len(df_source) != len(decoded_states):
            raise ValueError("Dataframe length and states length must match.")

        if len(decoded_states) == 0:
            return pd.DataFrame(columns=[
                "spell_id", "state", "start_frame", "end_frame",
                "start_timestamp", "end_timestamp", "duration_frames", "duration_seconds"
            ])

        state_changes = np.diff(decoded_states) != 0
        split_indices = np.where(state_changes)[0] + 1

        start_indices = np.insert(split_indices, 0, 0)
        end_indices = np.append(split_indices - 1, len(decoded_states) - 1)

        frame_ids = df_source["frame_id"].to_numpy()
        timestamps = df_source["timestamp"].to_numpy()

        spells: List[Dict] = []
        for spell_id, (s_idx, e_idx) in enumerate(zip(start_indices, end_indices)):
            state_int = int(decoded_states[s_idx])
            start_ts = float(timestamps[s_idx])
            end_ts = float(timestamps[e_idx])
            duration_frames = int(e_idx - s_idx + 1)

            spells.append({
                "spell_id": spell_id,
                "state": STATE_MAP.get(state_int, "UNKNOWN"),
                "start_frame": int(frame_ids[s_idx]),
                "end_frame": int(frame_ids[e_idx]),
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "duration_frames": duration_frames,
                "duration_seconds": np.around(end_ts - start_ts, decimals=4)
            })

        return pd.DataFrame(spells)
