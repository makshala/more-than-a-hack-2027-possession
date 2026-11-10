"""
Temporal diagnostics, transition statistics, and comparison utilities.
Addresses tasks 4, 6, and 7.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from config import STATE_MAP


class TemporalDiagnostics:
    """Analyzes spell metrics and evaluates decoding quality against ground truth and argmax."""

    @staticmethod
    def compute_diagnostics(
        df_source: pd.DataFrame,
        decoded_states: np.ndarray,
        df_spells: pd.DataFrame
    ) -> Dict[str, Any]:
        raw_argmax = np.argmax(df_source[["p_home", "p_away", "p_dead_ball"]].to_numpy(), axis=1)
        divergence = float(np.mean(raw_argmax != decoded_states))

        total_frames = len(decoded_states)
        distribution = {
            name: float(np.mean(decoded_states == s_id))
            for s_id, name in STATE_MAP.items()
        }

        # Spell transition statistics
        spell_seq = df_spells["state"].to_numpy()
        transitions = {}
        for src, dst in zip(spell_seq[:-1], spell_seq[1:]):
            key = f"{src} -> {dst}"
            transitions[key] = transitions.get(key, 0) + 1

        accuracy = None
        if "ground_truth" in df_source.columns:
            accuracy = float(np.mean(df_source["ground_truth"].to_numpy() == decoded_states))

        return {
            "total_frames": total_frames,
            "total_spells": len(df_spells),
            "state_distribution": distribution,
            "divergence_from_argmax": divergence,
            "accuracy_vs_ground_truth": accuracy,
            "mean_spell_duration_seconds": float(df_spells["duration_seconds"].mean()) if not df_spells.empty else 0.0,
            "min_spell_duration_seconds": float(df_spells["duration_seconds"].min()) if not df_spells.empty else 0.0,
            "spell_transitions": transitions
        }
