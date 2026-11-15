"""
Dynamic, Context-Aware HMM + Viterbi Decoder for 3-state possession estimation.
Executes exact Bayesian inversion on upstream posterior probabilities to recover true
emission log-likelihoods, preventing prior double-counting and temporal lag at spell boundaries.
"""
from typing import Optional, Literal, Callable, Union
import numpy as np
import pandas as pd
from config import (
    LOG_EPS, MIN_LOG_PROB, STATE_HOME, STATE_AWAY, STATE_DEAD_BALL,
    safe_log, safe_log_sum_exp
)


class ViterbiTemporalDecoder:
    """
    Log-domain Viterbi decoder with:
    1. Dynamic, context-aware transition matrices modulated by temporal entropy & contest dynamics.
    2. Exact Bayesian inversion converting raw upstream posteriors P(S|X) into true emissions P(X|S),
       eliminating double-counting of state priors and eradicating boundary temporal lag.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        prior_state_probs: Optional[np.ndarray] = None,
        input_semantics: Literal["posterior", "emission", "log_emission"] = "posterior",
        prior_temperature: float = 1.0,
        entropy_factor: float = 0.5,
        direct_turnover_penalty: float = 1.5,
        eps: float = LOG_EPS
    ):
        """
        Args:
            transition_matrix: Base (3, 3) state transition probability matrix.
            prior_state_probs: Prior distribution over states P(S) = [p_H, p_A, p_D].
            input_semantics: 'posterior' (applies Bayesian inversion), 'emission', or 'log_emission'.
            prior_temperature: Temperature scaling factor tau for Bayesian inversion.
            entropy_factor: Sensitivity to local prediction entropy for dynamic transition modulation.
            direct_turnover_penalty: Log-penalty applied to direct H <-> A transitions without dead-ball.
            eps: Epsilon threshold for numerical safety.
        """
        self.A_base = np.asarray(transition_matrix, dtype=np.float64)
        if self.A_base.shape != (3, 3):
            raise ValueError(f"Transition matrix must have shape (3, 3), got {self.A_base.shape}")

        # Ensure base transition matrix is strictly normalized
        row_sums = np.sum(self.A_base, axis=1, keepdims=True)
        self.A_base = self.A_base / np.maximum(row_sums, eps)

        self.eps = eps
        self.input_semantics = input_semantics
        self.prior_temperature = max(1e-3, float(prior_temperature))
        self.entropy_factor = float(entropy_factor)
        self.direct_turnover_penalty = float(direct_turnover_penalty)

        # Baseline prior state probabilities P(S)
        if prior_state_probs is None:
            # Empirical football default: Home 38%, Away 38%, Dead Ball 24%
            priors = np.array([0.38, 0.38, 0.24], dtype=np.float64)
        else:
            priors = np.asarray(prior_state_probs, dtype=np.float64)
            priors = priors / np.sum(priors)

        self.prior_state_probs = priors
        self.log_prior = safe_log(self.prior_state_probs, eps=self.eps)

    def _convert_to_log_emissions(self, df_probs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts probabilities and executes the Bayesian inversion sequence:
            log P(X|S) = log P(S|X) - (1 / tau) * log P(S)
        Returns (log_emissions, raw_posteriors).
        Supports both 'p_dead' (Role 4 standard) and 'p_dead_ball'.
        """
        dead_col = "p_dead_ball" if "p_dead_ball" in df_probs.columns else "p_dead"
        required_cols = ["p_home", "p_away", dead_col]
        if not all(col in df_probs.columns for col in required_cols):
            raise KeyError(f"DataFrame must contain columns: ['p_home', 'p_away', 'p_dead' (or 'p_dead_ball')]. Got {list(df_probs.columns)}")

        raw_scores = df_probs[["p_home", "p_away", dead_col]].to_numpy(dtype=np.float64)
        raw_scores = np.clip(raw_scores, self.eps, 1.0)
        # Normalize rows to ensure probabilities sum to 1
        raw_scores /= np.sum(raw_scores, axis=1, keepdims=True)

        if self.input_semantics == "posterior":
            # Exact Bayesian inversion:
            # log P(X|S) = log P(S|X) - (1 / tau) * log P(S)
            log_p_sx = safe_log(raw_scores, eps=self.eps)
            log_emissions = log_p_sx - (self.log_prior / self.prior_temperature)
            return log_emissions, raw_scores

        elif self.input_semantics == "emission":
            # Input is already likelihood P(X|S)
            log_emissions = safe_log(raw_scores, eps=self.eps)
            return log_emissions, raw_scores

        elif self.input_semantics == "log_emission":
            # Input is already log-likelihood
            return raw_scores, np.exp(np.clip(raw_scores, -700.0, 0.0))

        else:
            raise ValueError(f"Unknown input semantics: {self.input_semantics}")

    def compute_dynamic_transition_matrices(
        self,
        raw_posteriors: np.ndarray,
        context_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generates dynamic, context-aware transition matrices for each time step:
        log_A_seq has shape (T - 1, 3, 3) where log_A_seq[t, i, j] = log P(S_{t+1}=j | S_t=i).

        Modulations:
        1. Local prediction entropy H(t): When uncertainty/scramble is high, stickiness
           decreases and transition probability to Dead Ball (contest) increases organically.
        2. Direct turnover penalty: In tactical football, direct H -> A or A -> H turnovers
           without an intermediate contest or stoppage are physically rare.
        """
        T = len(raw_posteriors)
        if T <= 1:
            return np.empty((0, 3, 3), dtype=np.float64)

        # Compute Shannon entropy at each frame: H(t) = -sum p * log(p)
        clipped_p = np.clip(raw_posteriors, self.eps, 1.0)
        entropy = -np.sum(clipped_p * np.log(clipped_p), axis=1)  # Shape (T,)
        max_entropy = np.log(3.0)
        normalized_entropy = np.clip(entropy / max_entropy, 0.0, 1.0)  # Range [0, 1]

        # Base log transition matrix
        base_log_A = safe_log(self.A_base, eps=self.eps)  # Shape (3, 3)

        # Vectorized dynamic modulation across all t in [0, T - 2]
        # ent_t shape (T - 1, 1, 1)
        ent_t = normalized_entropy[:-1, np.newaxis, np.newaxis]

        # Broadcast base log A to (T - 1, 3, 3)
        log_A_dynamic = np.broadcast_to(base_log_A, (T - 1, 3, 3)).copy()

        # 1. Modulate stickiness by entropy: high entropy reduces self-transition
        if self.entropy_factor > 0:
            diag_idx = np.arange(3)
            # Reduce diagonal score proportionally to entropy
            log_A_dynamic[:, diag_idx, diag_idx] -= self.entropy_factor * ent_t[:, 0, 0][:, np.newaxis]
            # Boost transition to Dead Ball (state 2) when entropy is high
            log_A_dynamic[:, 0, STATE_DEAD_BALL] += 0.5 * self.entropy_factor * ent_t[:, 0, 0]
            log_A_dynamic[:, 1, STATE_DEAD_BALL] += 0.5 * self.entropy_factor * ent_t[:, 0, 0]

        # 2. Direct turnover physical constraint: Penalize direct H <-> A transitions
        if self.direct_turnover_penalty > 0:
            log_A_dynamic[:, STATE_HOME, STATE_AWAY] -= self.direct_turnover_penalty
            log_A_dynamic[:, STATE_AWAY, STATE_HOME] -= self.direct_turnover_penalty

        # Renormalize log-probabilities row-wise via log-sum-exp
        row_log_sums = safe_log_sum_exp(log_A_dynamic, axis=-1)  # Shape (T - 1, 3)
        log_A_dynamic -= row_log_sums[:, :, np.newaxis]

        return log_A_dynamic

    def decode(
        self,
        df_probs: pd.DataFrame,
        context_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Executes the exact dynamic log-Viterbi decoding algorithm.
        Returns:
            np.ndarray of decoded state sequence [0, 1, 2] of length T.
        """
        log_emissions, raw_posteriors = self._convert_to_log_emissions(df_probs)
        num_frames, num_states = log_emissions.shape

        if num_frames == 0:
            return np.empty(0, dtype=np.int32)

        # 1. Compute dynamic transition matrices: shape (T - 1, 3, 3)
        # where log_A_seq[t, i, j] is log P(S_{t+1}=j | S_t=i)
        log_A_seq = self.compute_dynamic_transition_matrices(
            raw_posteriors, context_features=context_features
        )

        viterbi_trellis = np.zeros((num_frames, num_states), dtype=np.float64)
        backpointers = np.zeros((num_frames, num_states), dtype=np.int32)

        # 2. Trellis Initialization (t = 0):
        # delta_0(s) = log P(S_0 = s) + log P(X_0 | S_0 = s)
        # With Bayesian inversion, log P(X_0 | S_0 = s) = log P(S_0=s | X_0) - log P(S_0=s),
        # so delta_0(s) identically equals log P(S_0 = s | X_0)!
        viterbi_trellis[0] = self.log_prior + log_emissions[0]

        # 3. Dynamic Forward Recursion (t = 1 ... T - 1)
        for t in range(1, num_frames):
            # log_A_seq[t - 1] has shape (3, 3): row i = source state, col j = dest state
            # Transpose so row j = dest state, col i = source state: log_A_T[j, i]
            log_A_T = log_A_seq[t - 1].T

            # scores[j, i] = delta_{t-1}(i) + log_A(i, j)
            transition_scores = viterbi_trellis[t - 1] + log_A_T  # Shape (3, 3)

            backpointers[t] = np.argmax(transition_scores, axis=1)
            viterbi_trellis[t] = np.max(transition_scores, axis=1) + log_emissions[t]

        # 4. Optimal Path Reconstruction (Backtracking)
        best_path = np.zeros(num_frames, dtype=np.int32)
        best_path[-1] = np.argmax(viterbi_trellis[-1])

        for t in range(num_frames - 2, -1, -1):
            best_path[t] = backpointers[t + 1, best_path[t + 1]]

        return best_path
