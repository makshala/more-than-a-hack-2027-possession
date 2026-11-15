"""
Explicit-Duration Hidden Semi-Markov Model (HSMM) decoder for possession phase estimation.
Models state durations using continuous parametric probability density functions (Gamma / Poisson)
fitted to historical tactical football data. Integrates continuous duration penalties directly into
the dynamic programming trellis to organically suppress micro-possessions and eliminate false positive boundaries.
"""
from typing import Optional, Dict, Literal, Tuple, Union
import numpy as np
import pandas as pd
from scipy.special import gammaln
from config import (
    LOG_EPS, MIN_LOG_PROB, DEFAULT_FPS, STATE_HOME, STATE_AWAY, STATE_DEAD_BALL,
    ms_to_frames, frames_to_ms, safe_log, TACTICAL_DURATION_PRIORS_MS,
    PHYSICAL_MIN_DURATIONS_MS, MAX_SEARCH_DURATION_MS
)


class ContinuousDurationDistribution:
    """
    Parametric continuous duration density function for tactical football possession.
    Supports continuous Gamma and Poisson distributions with log-domain evaluation.
    """

    def __init__(
        self,
        dist_type: Literal["gamma", "poisson"] = "gamma",
        mean_ms: float = 11500.0,
        std_ms: float = 7800.0,
        shape_k: Optional[float] = None,
        scale_theta: Optional[float] = None,
        lambda_val: Optional[float] = None,
        fps: float = DEFAULT_FPS,
        eps: float = LOG_EPS
    ):
        self.dist_type = dist_type
        self.fps = fps
        self.eps = eps

        if dist_type == "gamma":
            # Parametrize Gamma: shape k > 1 guarantees f(0) = 0 and soft suppression of micro-spells
            if shape_k is not None and scale_theta is not None:
                self.k = float(shape_k)
                self.theta = float(scale_theta)
            else:
                mean_sec = mean_ms / 1000.0
                std_sec = max(1.0, std_ms / 1000.0)
                # Method of moments for Gamma:
                # mean = k * theta, var = k * theta^2
                # => theta = var / mean, k = (mean / std)^2
                self.k = max(1.2, float((mean_sec / std_sec) ** 2))
                self.theta = max(0.5, float((std_sec ** 2) / mean_sec))

            # Precompute log-normalizing constant: -log(Gamma(k)) - k * log(theta)
            self._log_norm = -float(gammaln(self.k)) - self.k * float(np.log(self.theta))

        elif dist_type == "poisson":
            if lambda_val is not None:
                self.lambda_val = float(lambda_val)
            else:
                # Mean in frames
                self.lambda_val = max(5.0, float((mean_ms / 1000.0) * fps))

        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    def log_pdf_frames(self, duration_frames: np.ndarray | int) -> np.ndarray | float:
        """
        Evaluates log-probability density for a given duration in frames.
        Continuous time evaluation in seconds: t = duration_frames / fps.
        """
        d = np.asarray(duration_frames, dtype=np.float64)
        t_sec = np.maximum(d / self.fps, 1e-4)

        if self.dist_type == "gamma":
            # Continuous Gamma log-pdf:
            # log f(t; k, theta) = (k - 1) * log(t) - (t / theta) + log_norm
            # Note: Jacobian adjustment |dt/dd| = 1/fps is constant across d and absorbed or added
            log_density = (self.k - 1.0) * np.log(t_sec) - (t_sec / self.theta) + self._log_norm - np.log(self.fps)
            return log_density

        elif self.dist_type == "poisson":
            # Shifted Poisson log-pmf: log p(d; lambda) = (d - 1)*log(lambda) - lambda - log(Gamma(d))
            d_shifted = np.maximum(d - 1.0, 0.0)
            log_pmf = d_shifted * np.log(self.lambda_val) - self.lambda_val - gammaln(np.maximum(d, 1.0))
            return log_pmf


class HSMMExplicitDurationDecoder:
    """
    Explicit-duration Hidden Semi-Markov Model (HSMM) decoder.
    Integrates parametric duration log-density penalties directly into the dynamic
    programming trellis, organically suppressing physically impossible micro-possessions
    while rewarding sustained tactical possession sequences.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        duration_distributions: Optional[Dict[int, ContinuousDurationDistribution]] = None,
        duration_weight: float = 1.0,
        prior_state_probs: Optional[np.ndarray] = None,
        input_semantics: Literal["posterior", "emission", "log_emission"] = "posterior",
        prior_temperature: float = 1.0,
        min_duration_ms: Optional[Dict[int, float]] = None,
        max_duration_ms: float = MAX_SEARCH_DURATION_MS,
        fps: float = DEFAULT_FPS,
        eps: float = LOG_EPS
    ):
        """
        Args:
            transition_matrix: (3, 3) state transition probability matrix.
            duration_distributions: Dict mapping state_id (0, 1, 2) to ContinuousDurationDistribution.
            duration_weight: Weight omega_d on continuous duration log-likelihood penalty in trellis.
            prior_state_probs: Prior state distribution P(S).
            input_semantics: 'posterior' (Bayesian inversion), 'emission', or 'log_emission'.
            prior_temperature: Temperature scaling factor tau for Bayesian inversion.
            min_duration_ms: Physical lower bound in ms for trellis search window.
            max_duration_ms: Physical upper bound in ms for trellis search window.
            fps: Video telemetry frame rate.
            eps: Epsilon threshold for numerical safety.
        """
        self.fps = fps
        self.eps = eps
        self.duration_weight = float(duration_weight)
        self.input_semantics = input_semantics
        self.prior_temperature = max(1e-3, float(prior_temperature))

        # Base transition matrix:
        # In HSMM, transitions between states must not include self-transitions
        # (self-transitions are governed by the explicit duration distribution).
        A_raw = np.asarray(transition_matrix, dtype=np.float64)
        if A_raw.shape != (3, 3):
            raise ValueError(f"Expected transition matrix of shape (3, 3), got {A_raw.shape}")

        A_off = A_raw.copy()
        np.fill_diagonal(A_off, 0.0)
        row_sums = np.sum(A_off, axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums <= 0] = 1.0
        self.A = A_off / row_sums
        self.log_A = safe_log(self.A, eps=self.eps)

        # Baseline state priors
        if prior_state_probs is None:
            priors = np.array([0.38, 0.38, 0.24], dtype=np.float64)
        else:
            priors = np.asarray(prior_state_probs, dtype=np.float64)
            priors = priors / np.sum(priors)
        self.prior_state_probs = priors
        self.log_prior = safe_log(self.prior_state_probs, eps=self.eps)

        # Initialize continuous duration distributions
        if duration_distributions is not None:
            self.duration_dists = duration_distributions
        else:
            self.duration_dists = {}
            for s in [STATE_HOME, STATE_AWAY, STATE_DEAD_BALL]:
                p = TACTICAL_DURATION_PRIORS_MS[s]
                self.duration_dists[s] = ContinuousDurationDistribution(
                    dist_type="gamma",
                    mean_ms=p["mean_ms"],
                    std_ms=p["std_ms"],
                    fps=self.fps,
                    eps=self.eps
                )

        # Physical search window limits
        min_ms_map = min_duration_ms if min_duration_ms is not None else PHYSICAL_MIN_DURATIONS_MS
        self.min_frames = {s: ms_to_frames(min_ms_map.get(s, 400.0), self.fps) for s in range(3)}
        self.max_search_frames = ms_to_frames(max_duration_ms, self.fps)

        # Precompute duration log-penalties up to max_search_frames for fast trellis evaluation
        self._precomputed_duration_logp: Dict[int, np.ndarray] = {}
        for s in range(3):
            d_range = np.arange(1, self.max_search_frames + 1, dtype=np.float64)
            log_p = self.duration_dists[s].log_pdf_frames(d_range)
            self._precomputed_duration_logp[s] = np.insert(log_p, 0, MIN_LOG_PROB)

    def _convert_to_log_emissions(self, df_probs: pd.DataFrame) -> np.ndarray:
        """Extracts probabilities and applies Bayesian inversion. Supports 'p_dead' and 'p_dead_ball'."""
        dead_col = "p_dead_ball" if "p_dead_ball" in df_probs.columns else "p_dead"
        required_cols = ["p_home", "p_away", dead_col]
        if not all(col in df_probs.columns for col in required_cols):
            raise KeyError(f"DataFrame must contain columns: ['p_home', 'p_away', 'p_dead' (or 'p_dead_ball')]. Got {list(df_probs.columns)}")

        raw_scores = df_probs[["p_home", "p_away", dead_col]].to_numpy(dtype=np.float64)
        raw_scores = np.clip(raw_scores, self.eps, 1.0)
        raw_scores /= np.sum(raw_scores, axis=1, keepdims=True)

        if self.input_semantics == "posterior":
            # Exact Bayesian inversion
            log_emissions = safe_log(raw_scores, eps=self.eps) - (self.log_prior / self.prior_temperature)
            return log_emissions
        elif self.input_semantics == "emission":
            return safe_log(raw_scores, eps=self.eps)
        elif self.input_semantics == "log_emission":
            return raw_scores
        else:
            raise ValueError(f"Unknown input semantics: {self.input_semantics}")

    def decode(self, df_probs: pd.DataFrame) -> np.ndarray:
        """
        Executes the explicit-duration semi-Markov dynamic programming trellis.
        Optimally segments the time series into spells by incorporating continuous duration
        log-density penalties.
        """
        log_emissions = self._convert_to_log_emissions(df_probs)
        T, num_states = log_emissions.shape

        if T == 0:
            return np.empty(0, dtype=np.int32)

        # Prefix sums of log emissions for O(1) segment emission evaluation:
        # segment sum log P(X_{t-d+1:t} | S=s) = prefix[t, s] - prefix[t - d, s]
        prefix_emissions = np.vstack([
            np.zeros((1, num_states), dtype=np.float64),
            np.cumsum(log_emissions, axis=0)
        ])

        # dp[t, s] = max score of valid segmentation up to frame t ending with state s
        dp = np.full((T + 1, num_states), MIN_LOG_PROB, dtype=np.float64)
        # backpointers[t, s] = [prev_state, duration_d]
        backpointers = np.zeros((T + 1, num_states, 2), dtype=np.int32)

        # Initialization at t = 0 (boundary)
        for s in range(num_states):
            min_d = self.min_frames[s]
            max_d = min(self.max_search_frames, T)
            d_vals = np.arange(min_d, max_d + 1)
            if len(d_vals) == 0:
                continue

            seg_emissions = prefix_emissions[d_vals, s] - prefix_emissions[0, s]
            duration_logp = self._precomputed_duration_logp[s][d_vals]
            total_scores = self.log_prior[s] + (self.duration_weight * duration_logp) + seg_emissions

            for idx, d in enumerate(d_vals):
                if total_scores[idx] > dp[d, s]:
                    dp[d, s] = total_scores[idx]
                    backpointers[d, s] = [-1, d]

        # Dynamic programming forward pass over time frames
        for t in range(1, T + 1):
            for s in range(num_states):
                min_d = self.min_frames[s]
                max_d = min(self.max_search_frames, t)
                if min_d > max_d:
                    continue

                d_vals = np.arange(min_d, max_d + 1)
                t_prev = t - d_vals  # Array of start frames

                # Segment emission scores for all candidate durations
                seg_emissions = prefix_emissions[t, s] - prefix_emissions[t_prev, s]
                duration_logp = self._precomputed_duration_logp[s][d_vals]

                # Max over previous states s' != s
                other_states = [sp for sp in range(num_states) if sp != s]
                # dp[t_prev, other_states] shape: (len(d_vals), len(other_states))
                prev_scores = dp[t_prev][:, other_states] + self.log_A[other_states, s]
                best_prev_idx = np.argmax(prev_scores, axis=1)
                best_prev_score = prev_scores[np.arange(len(d_vals)), best_prev_idx]
                best_prev_state = np.array(other_states)[best_prev_idx]

                candidate_scores = best_prev_score + (self.duration_weight * duration_logp) + seg_emissions

                best_d_idx = np.argmax(candidate_scores)
                if candidate_scores[best_d_idx] > dp[t, s]:
                    dp[t, s] = candidate_scores[best_d_idx]
                    backpointers[t, s] = [best_prev_state[best_d_idx], d_vals[best_d_idx]]

        # Backtracking optimal state sequence
        best_end_state = int(np.argmax(dp[T]))
        decoded_path = np.zeros(T, dtype=np.int32)

        curr_t = T
        curr_state = best_end_state

        while curr_t > 0:
            prev_state, d = backpointers[curr_t, curr_state]
            if d <= 0:
                # Fallback safeguard in degenerate edge cases
                d = max(1, curr_t)
            start_t = max(0, curr_t - d)
            decoded_path[start_t:curr_t] = curr_state
            curr_t = start_t
            curr_state = prev_state

        return decoded_path
