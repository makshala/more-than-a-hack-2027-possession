"""
Synthetic possession telemetry data generator.
Addresses task 3: Synthetic sequences with controlled H/A/D transitions.
"""
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from config import DEFAULT_FPS


class SyntheticPossessionGenerator:
    """Generates synthetic frame-level sequences mimicking Role 4 model outputs."""

    @staticmethod
    def generate_controlled_scenario(
        segments: List[Tuple[int, int]],
        fps: float = DEFAULT_FPS,
        base_confidence: float = 0.85,
        noise_level: float = 0.15,
        random_seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """
        Generates synthetic frames based on predefined segments.
        segments: List of tuples (state_id, num_frames)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        ground_truth_states = []
        for state, count in segments:
            ground_truth_states.extend([state] * count)
        
        ground_truth = np.array(ground_truth_states, dtype=np.int32)
        total_frames = len(ground_truth)
        
        dt = 1.0 / fps
        timestamps = np.around(np.arange(total_frames) * dt, decimals=4)
        frame_ids = np.arange(total_frames, dtype=np.int64)
        
        probs = np.zeros((total_frames, 3), dtype=np.float64)
        
        for t in range(total_frames):
            target_state = ground_truth[t]
            base = np.full(3, (1.0 - base_confidence) / 2.0, dtype=np.float64)
            base[target_state] = base_confidence
            
            # Dirichlet noise keeping sum to 1
            alpha = base * 25.0 + 1e-2
            sample = np.random.dirichlet(alpha)
            
            # Isolated single-frame flicker / measurement artifact (noise)
            if np.random.rand() < noise_level:
                sample = np.random.dirichlet([1.0, 1.0, 1.0])
                
            probs[t] = sample
            
        return pd.DataFrame({
            "frame_id": frame_ids,
            "timestamp": timestamps,
            "p_home": probs[:, 0],
            "p_away": probs[:, 1],
            "p_dead_ball": probs[:, 2],
            "ground_truth": ground_truth
        })
