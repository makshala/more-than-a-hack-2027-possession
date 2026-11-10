"""
HMM + Viterbi Decoder for 3-state possession estimation.
Addresses tasks 1, 2, and 8 (posterior / log-likelihood semantic handling).
"""
from typing import Optional, Literal
import numpy as np
import pandas as pd
from config import EPS


class ViterbiTemporalDecoder:
    """Viterbi decoder operating in log-domain with input semantic conversion."""

    def __init__(
        self,
        transition_matrix: np.ndarray,
        prior_state_probs: Optional[np.ndarray] = None,
        input_semantics: Literal["posterior", "emission", "log_emission"] = "posterior",
        eps: float = EPS
    ):
        self.A = np.asarray(transition_matrix, dtype=np.float64)
        if self.A.shape != (3, 3):
            raise ValueError(f"Transition matrix must have shape (3, 3), got {self.A.shape}")

        self.eps = eps
        self.log_A = np.log(np.clip(self.A, self.eps, 1.0))
        self.input_semantics = input_semantics

        if prior_state_probs is None:
            self.log_prior = np.log(np.full(3, 1.0 / 3.0, dtype=np.float64))
        else:
            priors = np.asarray(prior_state_probs, dtype=np.float64)
            priors = priors / np.sum(priors)
            self.log_prior = np.log(np.clip(priors, self.eps, 1.0))

    def _convert_to_log_emissions(self, df_probs: pd.DataFrame) -> np.ndarray:
        required_cols = ["p_home", "p_away", "p_dead_ball"]
        if not all(col in df_probs.columns for col in required_cols):
            raise KeyError(f"DataFrame must contain columns: {required_cols}")

        raw_scores = df_probs[required_cols].to_numpy(dtype=np.float64)

        if self.input_semantics == "posterior":
            # Bayes inversion: log P(X|S) = log P(S|X) - log P(S)
            return np.log(np.clip(raw_scores, self.eps, 1.0)) - self.log_prior
        elif self.input_semantics == "emission":
            # Direct likelihood: log P(X|S)
            return np.log(np.clip(raw_scores, self.eps, 1.0))
        elif self.input_semantics == "log_emission":
            return raw_scores
        else:
            raise ValueError(f"Unknown input semantics: {self.input_semantics}")

    def decode(self, df_probs: pd.DataFrame) -> np.ndarray:
        log_emissions = self._convert_to_log_emissions(df_probs)
        num_frames, num_states = log_emissions.shape

        if num_frames == 0:
            return np.empty(0, dtype=np.int32)

        viterbi_trellis = np.zeros((num_frames, num_states), dtype=np.float64)
        backpointers = np.zeros((num_frames, num_states), dtype=np.int32)

        viterbi_trellis[0] = self.log_prior + log_emissions[0]
        log_A_T = self.log_A.T

        for t in range(1, num_frames):
            scores = viterbi_trellis[t - 1] + log_A_T
            backpointers[t] = np.argmax(scores, axis=1)
            viterbi_trellis[t] = np.max(scores, axis=1) + log_emissions[t]

        best_path = np.zeros(num_frames, dtype=np.int32)
        best_path[-1] = np.argmax(viterbi_trellis[-1])

        for t in range(num_frames - 2, -1, -1):
            best_path[t] = backpointers[t + 1, best_path[t + 1]]

        return best_path
