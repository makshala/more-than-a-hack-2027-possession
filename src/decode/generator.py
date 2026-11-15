"""
Synthetic possession telemetry data generator.
Models empirically derived tracking error distributions and bursty Markovian noise regimes
simulating real-world football computer vision failures (occlusions, player clustering, camera cuts).
"""
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
from config import (
    DEFAULT_FPS, STATE_HOME, STATE_AWAY, STATE_DEAD_BALL,
    ms_to_frames, TACTICAL_DURATION_PRIORS_MS
)


# Markovian noise regimes
REGIME_CLEAN: int = 0
REGIME_OCCLUSION: int = 1
REGIME_CLUSTERING: int = 2
REGIME_CAMERA_CUT: int = 3

REGIME_NAMES: Dict[int, str] = {
    REGIME_CLEAN: "Clean Tracking",
    REGIME_OCCLUSION: "Player Occlusion",
    REGIME_CLUSTERING: "Player Clustering / Scramble",
    REGIME_CAMERA_CUT: "Broadcast Camera Cut"
}


class SyntheticPossessionGenerator:
    """
    Generates realistic synthetic frame-level sequences mimicking upstream ML model outputs.
    Abandons naive uniform noise in favor of empirically derived tracking error distributions
    and bursty Markovian noise regimes.
    """

    # Transition probability matrix for Markovian noise regimes:
    # State order: [Clean, Occlusion, Clustering, Camera Cut]
    # Strong diagonal persistence produces realistic multi-frame burst errors.
    DEFAULT_REGIME_TRANSITIONS: np.ndarray = np.array([
        [0.965, 0.015, 0.015, 0.005],  # From Clean
        [0.120, 0.880, 0.000, 0.000],  # From Occlusion (avg burst ~8.3 frames)
        [0.100, 0.000, 0.900, 0.000],  # From Clustering (avg burst ~10 frames)
        [0.080, 0.000, 0.000, 0.920],  # From Camera Cut (avg burst ~12.5 frames)
    ], dtype=np.float64)

    @classmethod
    def _sample_markov_regimes(
        cls,
        total_frames: int,
        transition_matrix: Optional[np.ndarray] = None,
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """Generates a sequence of bursty noise regimes via Markov chain simulation."""
        rng = random_state if random_state is not None else np.random.RandomState()
        T = transition_matrix if transition_matrix is not None else cls.DEFAULT_REGIME_TRANSITIONS
        
        regimes = np.zeros(total_frames, dtype=np.int32)
        current = REGIME_CLEAN
        
        for t in range(total_frames):
            regimes[t] = current
            # Transition to next regime
            p_next = T[current]
            current = rng.choice(len(p_next), p=p_next)
            
        return regimes

    @classmethod
    def _sample_emission_probabilities(
        cls,
        ground_truth: np.ndarray,
        regimes: np.ndarray,
        base_confidence: float = 0.88,
        rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Samples upstream model posterior probabilities P(S|X) from empirically derived
        Dirichlet tracking error distributions tailored to each noise regime.
        """
        if rng is None:
            rng = np.random.RandomState()

        total_frames = len(ground_truth)
        probs = np.zeros((total_frames, 3), dtype=np.float64)

        for t in range(total_frames):
            gt_state = ground_truth[t]
            regime = regimes[t]

            if regime == REGIME_CLEAN:
                # Clean tracking: High confidence with log-normal Dirichlet dispersion
                dispersion = rng.lognormal(mean=3.2, sigma=0.35)
                base = np.full(3, (1.0 - base_confidence) / 2.0, dtype=np.float64)
                base[gt_state] = base_confidence
                alpha = np.maximum(base * dispersion, 1e-3)
                probs[t] = rng.dirichlet(alpha)

            elif regime == REGIME_OCCLUSION:
                # Occlusion event: Ball carrier or ball is occluded.
                # Entropy spikes; probabilities diffuse toward flat Dirichlet distribution
                # with high variance and occasional spurious dead-ball classification.
                alpha = np.array([1.2, 1.2, 1.8], dtype=np.float64)
                probs[t] = rng.dirichlet(alpha)

            elif regime == REGIME_CLUSTERING:
                # Player clustering / midfield scramble:
                # Multiple players contesting ball creates severe Home vs Away confusion (cross-talk).
                # High entropy between H and A, low D probability.
                concentration = rng.uniform(8.0, 18.0)
                if gt_state in (STATE_HOME, STATE_AWAY):
                    # Ambiguous duel between Home and Away
                    alpha = np.array([0.48 * concentration, 0.48 * concentration, 0.04 * concentration])
                else:
                    alpha = np.array([0.35 * concentration, 0.35 * concentration, 0.30 * concentration])
                probs[t] = rng.dirichlet(alpha)

            elif regime == REGIME_CAMERA_CUT:
                # Broadcast camera cut / replay:
                # Sudden discontinuity in visual feed. Upstream model often defaults to
                # dead-ball or flat uncertain distribution.
                cut_type = rng.rand()
                if cut_type < 0.70:
                    # Spurious dead ball spike
                    alpha = np.array([0.5, 0.5, 8.0], dtype=np.float64)
                else:
                    # Total visual blackout / uniform uncertainty
                    alpha = np.array([1.0, 1.0, 1.0], dtype=np.float64)
                probs[t] = rng.dirichlet(alpha)

        # Numerical safety: clip and normalize to guarantee strict sum-to-1
        probs = np.clip(probs, 1e-12, 1.0)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    @classmethod
    def generate_controlled_scenario(
        cls,
        segments: List[Tuple[int, int]],
        fps: float = DEFAULT_FPS,
        base_confidence: float = 0.88,
        noise_regimes: Optional[np.ndarray] = None,
        random_seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """
        Generates synthetic frames based on predefined tactical segments.
        segments: List of tuples (state_id, num_frames)
        """
        rng = np.random.RandomState(random_seed)

        ground_truth_states: List[int] = []
        for state, count in segments:
            ground_truth_states.extend([state] * count)

        ground_truth = np.array(ground_truth_states, dtype=np.int32)
        total_frames = len(ground_truth)

        dt = 1.0 / fps
        timestamps = np.around(np.arange(total_frames) * dt, decimals=4)
        frame_ids = np.arange(total_frames, dtype=np.int64)

        if noise_regimes is None:
            regimes = cls._sample_markov_regimes(total_frames, random_state=rng)
        else:
            regimes = np.asarray(noise_regimes, dtype=np.int32)

        probs = cls._sample_emission_probabilities(
            ground_truth, regimes, base_confidence=base_confidence, rng=rng
        )

        return pd.DataFrame({
            "frame_id": frame_ids,
            "timestamp": timestamps,
            "p_home": probs[:, 0],
            "p_away": probs[:, 1],
            "p_dead_ball": probs[:, 2],
            "ground_truth": ground_truth,
            "noise_regime": regimes
        })

    @classmethod
    def generate_tactical_match_simulation(
        cls,
        num_spells: int = 25,
        fps: float = DEFAULT_FPS,
        base_confidence: float = 0.88,
        random_seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """
        Generates a realistic tactical sequence where spell durations are sampled from
        continuous Gamma distributions fitted to professional football match data.
        Transitions follow tactical rules (e.g. turnovers usually go through dead-ball / contest).
        """
        rng = np.random.RandomState(random_seed)

        segments: List[Tuple[int, int]] = []
        current_state = rng.choice([STATE_HOME, STATE_AWAY])

        # Tactical transition routing: leaving H or A enters D 80% of the time
        p_turnover_to_dead_ball = 0.80

        for _ in range(num_spells):
            # Sample duration from Gamma distribution parameterized by mean and std
            prior = TACTICAL_DURATION_PRIORS_MS[current_state]
            mean_ms = prior["mean_ms"]
            std_ms = prior["std_ms"]
            
            # Gamma parameterization: shape k = (mean/std)^2, scale theta = std^2 / mean
            shape_k = (mean_ms / std_ms) ** 2
            scale_theta = (std_ms ** 2) / mean_ms
            duration_ms = rng.gamma(shape=shape_k, scale=scale_theta)
            duration_ms = np.clip(duration_ms, prior["min_ms"], prior["max_ms"])
            
            num_frames = ms_to_frames(duration_ms, fps)
            segments.append((current_state, num_frames))

            # Sample next tactical state
            if current_state == STATE_DEAD_BALL:
                # Dead ball resumes with possession to either Home or Away
                current_state = rng.choice([STATE_HOME, STATE_AWAY], p=[0.5, 0.5])
            else:
                opponent = STATE_AWAY if current_state == STATE_HOME else STATE_HOME
                if rng.rand() < p_turnover_to_dead_ball:
                    current_state = STATE_DEAD_BALL
                else:
                    current_state = opponent

        return cls.generate_controlled_scenario(
            segments=segments,
            fps=fps,
            base_confidence=base_confidence,
            random_seed=random_seed
        )
