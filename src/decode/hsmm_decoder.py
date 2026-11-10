"""
Duration-aware Explicit-Duration Hidden Markov Model (HSMM) decoder.
Addresses tasks 5 and 6: Minimum/preferred state duration & comparison with standard HMM.
"""
from typing import Optional, Dict, Literal
import numpy as np
import pandas as pd
from config import EPS


class HSMMExplicitDurationDecoder:
    """
    Explicit-duration semi-Markov decoder enforcing minimum and maximum state durations.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        min_durations: Dict[int, int],
        max_durations: Dict[int, int],
        prior_state_probs: Optional[np.ndarray] = None,
        input_semantics: Literal["posterior", "emission"] = "posterior",
        eps: float = EPS
    ):
        """
        min_durations: dict mapping state_id (0, 1, 2) to minimum duration in frames
        max_durations: dict mapping state_id (0, 1, 2) to maximum duration considered in frames
        """
        self.A = np.asarray(transition_matrix, dtype=np.float64)
        self.eps = eps
        self.log_A = np.log(np.clip(self.A, self.eps, 1.0))
        self.min_durations = min_durations
        self.max_durations = max_durations
        self.input_semantics = input_semantics

        if prior_state_probs is None:
            self.log_prior = np.log(np.full(3, 1.0 / 3.0, dtype=np.float64))
        else:
            priors = np.asarray(prior_state_probs, dtype=np.float64)
            priors = priors / np.sum(priors)
            self.log_prior = np.log(np.clip(priors, self.eps, 1.0))

    def decode(self, df_probs: pd.DataFrame) -> np.ndarray:
        required_cols = ["p_home", "p_away", "p_dead_ball"]
        raw_scores = df_probs[required_cols].to_numpy(dtype=np.float64)

        if self.input_semantics == "posterior":
            log_emissions = np.log(np.clip(raw_scores, self.eps, 1.0)) - self.log_prior
        else:
            log_emissions = np.log(np.clip(raw_scores, self.eps, 1.0))

        T, num_states = log_emissions.shape
        if T == 0:
            return np.empty(0, dtype=np.int32)

        # Prefix sums of log emissions for fast segment score computation
        prefix_emissions = np.vstack([np.zeros((1, num_states)), np.cumsum(log_emissions, axis=0)])

        # dp[t, s] = max score of valid segmentation up to frame t ending with state s
        dp = np.full((T + 1, num_states), -np.inf, dtype=np.float64)
        backpointers = np.zeros((T + 1, num_states, 2), dtype=np.int32)  # [prev_state, duration]

        # Initialization
        for s in range(num_states):
            min_d = self.min_durations.get(s, 1)
            max_d = min(self.max_durations.get(s, 50), T)
            for d in range(min_d, max_d + 1):
                seg_score = prefix_emissions[d, s] - prefix_emissions[0, s]
                total_score = self.log_prior[s] + seg_score
                if total_score > dp[d, s]:
                    dp[d, s] = total_score
                    backpointers[d, s] = [-1, d]

        # Dynamic programming over frames
        for t in range(1, T + 1):
            for s in range(num_states):
                min_d = self.min_durations.get(s, 1)
                max_d = min(self.max_durations.get(s, 50), t)
                for d in range(min_d, max_d + 1):
                    t_prev = t - d
                    seg_score = prefix_emissions[t, s] - prefix_emissions[t_prev, s]
                    for prev_s in range(num_states):
                        if prev_s == s:
                            continue
                        trans_score = dp[t_prev, prev_s] + self.log_A[prev_s, s] + seg_score
                        if trans_score > dp[t, s]:
                            dp[t, s] = trans_score
                            backpointers[t, s] = [prev_s, d]

        # Backtracking
        best_end_state = int(np.argmax(dp[T]))
        decoded_path = np.zeros(T, dtype=np.int32)

        curr_t = T
        curr_state = best_end_state

        while curr_t > 0:
            prev_state, d = backpointers[curr_t, curr_state]
            decoded_path[curr_t - d : curr_t] = curr_state
            curr_t -= d
            curr_state = prev_state

        return decoded_path
