"""
Configuration constants, physical possession constraints, numerical safety boundaries,
and hyperparameter search space definitions for possession temporal decoding.
"""
from typing import Dict, Any, Tuple, List
import numpy as np


# -----------------------------------------------------------------------------
# 1. Football Possession State Definitions
# -----------------------------------------------------------------------------
STATE_HOME: int = 0
STATE_AWAY: int = 1
STATE_DEAD_BALL: int = 2
NUM_STATES: int = 3

STATE_MAP: Dict[int, str] = {
    STATE_HOME: "H",
    STATE_AWAY: "A",
    STATE_DEAD_BALL: "D"
}

INV_STATE_MAP: Dict[str, int] = {
    "H": STATE_HOME,
    "A": STATE_AWAY,
    "D": STATE_DEAD_BALL
}

STATE_NAMES: Dict[str, str] = {
    "H": "Home Possession",
    "A": "Away Possession",
    "D": "Dead Ball / In Contest"
}

DEFAULT_FPS: float = 25.0

# Default data file paths for Role 4 inputs and Role 5 outputs
DEFAULT_PARQUET_PATH: str = "role5_probabilities.parquet"
DEFAULT_SYNC_PATH: str = "sync_offset.json"
DEFAULT_OUTPUT_SPELLS_PARQUET: str = "role5_decoded_spells.parquet"
DEFAULT_OUTPUT_SPELLS_CSV: str = "role5_decoded_spells.csv"


# -----------------------------------------------------------------------------
# 2. Temporal Conversion & Data Normalization Utilities
# -----------------------------------------------------------------------------
def ms_to_frames(ms: float, fps: float = DEFAULT_FPS) -> int:
    """Converts a duration in milliseconds to integer frame count at given fps."""
    return max(1, int(round((ms / 1000.0) * fps)))


def frames_to_ms(frames: int, fps: float = DEFAULT_FPS) -> float:
    """Converts a frame count to duration in milliseconds at given fps."""
    return float((frames / fps) * 1000.0)


def seconds_to_ms(seconds: float) -> float:
    """Converts seconds to milliseconds."""
    return float(seconds * 1000.0)


def ms_to_seconds(ms: float) -> float:
    """Converts milliseconds to seconds."""
    return float(ms / 1000.0)


def normalize_probabilities_df(df: Any, fps: float = DEFAULT_FPS) -> Any:
    """
    Normalizes input DataFrame from Role 4 or synthetic generators:
    1. Aliases p_dead <-> p_dead_ball so both naming conventions work everywhere.
    2. Ensures frame_id exists.
    3. Computes timestamp = frame_id / fps if missing.
    4. Clips and normalizes probabilities row-wise to guarantee strict sum-to-1.
    """
    import pandas as pd
    res = df.copy()

    # Column aliasing for Dead Ball
    if "p_dead" in res.columns and "p_dead_ball" not in res.columns:
        res["p_dead_ball"] = res["p_dead"]
    elif "p_dead_ball" in res.columns and "p_dead" not in res.columns:
        res["p_dead"] = res["p_dead_ball"]

    required_probs = ["p_home", "p_away", "p_dead_ball"]
    if not all(col in res.columns for col in required_probs):
        raise KeyError(
            f"DataFrame must contain probability columns 'p_home', 'p_away', and 'p_dead' (or 'p_dead_ball'). "
            f"Found: {list(res.columns)}"
        )

    # Frame ID & Timestamps
    if "frame_id" not in res.columns:
        res["frame_id"] = np.arange(len(res), dtype=np.int64)

    if "timestamp" not in res.columns:
        res["timestamp"] = np.around(res["frame_id"].to_numpy() / fps, decimals=4)

    # Ensure probabilities are normalized and strictly sum to 1
    probs = res[required_probs].to_numpy(dtype=np.float64)
    probs = np.clip(probs, LOG_EPS, 1.0)
    row_sums = np.sum(probs, axis=1, keepdims=True)
    probs /= row_sums
    res["p_home"] = probs[:, 0]
    res["p_away"] = probs[:, 1]
    res["p_dead_ball"] = probs[:, 2]
    res["p_dead"] = probs[:, 2]

    return res


# -----------------------------------------------------------------------------
# 3. Physical Possession Constraints (Stored in Absolute Milliseconds)
# -----------------------------------------------------------------------------
# In professional football, true possession transfers require ball control/touches
# which physically take at least 800-1000 ms. Sub-second flickers are tracking artifacts.
PHYSICAL_MIN_DURATIONS_MS: Dict[int, float] = {
    STATE_HOME: 600.0,        # było 1000.0 (15 klatek zamiast 25)
    STATE_AWAY: 600.0,        # było 1000.0
    STATE_DEAD_BALL: 480.0    # było 600.0
}

SANDWICH_MERGE_MAX_MS: float = 600.0  

# Empirical tactical football duration priors (mean and standard deviation in ms)
# Derived from elite professional match statistics (e.g. StatsBomb / Wyscout data).
TACTICAL_DURATION_PRIORS_MS: Dict[int, Dict[str, float]] = {
    STATE_HOME: {
        "mean_ms": 11500.0,     # ~11.5 seconds average spell
        "std_ms": 7800.0,       # High variance reflecting short turnovers vs long build-ups
        "min_ms": 600.0,
        "max_ms": 45000.0
    },
    STATE_AWAY: {
        "mean_ms": 11500.0,
        "std_ms": 7800.0,
        "min_ms": 600.0,
        "max_ms": 45000.0
    },
    STATE_DEAD_BALL: {
        "mean_ms": 17200.0,     # ~17.2 seconds average stoppage (throw-ins, free kicks, goal kicks)
        "std_ms": 11000.0,
        "min_ms": 400.0,
        "max_ms": 60000.0
    }
}

# Maximum duration window for dynamic programming search (ms)
MAX_SEARCH_DURATION_MS: float = 35000.0

# Morphological filtering temporal kernels (in absolute milliseconds)
DEFAULT_CLOSING_KERNEL_MS: float = 600.0    # Bridges tracking dropouts < 600ms
DEFAULT_OPENING_KERNEL_MS: float = 320.0    # Eliminates single/dual-frame blips < 320ms

# Hysteresis stability thresholds
HYSTERESIS_CERTAINTY_THRESHOLD: float = 0.60
HYSTERESIS_MARGIN_THRESHOLD: float = 0.20


# -----------------------------------------------------------------------------
# 4. Strict Logarithmic Epsilon and Numerical Underflow Boundaries
# -----------------------------------------------------------------------------
# Strict epsilon boundaries to entirely eliminate floating-point underflow
LOG_EPS: float = 1e-15
MIN_LOG_PROB: float = -1e9
MAX_LOG_PROB: float = 0.0
PROB_CLIP_MIN: float = 1e-12
PROB_CLIP_MAX: float = 1.0 - 1e-12
EPS: float = 1e-12  # Legacy backward compatibility


def safe_log(
    p: np.ndarray | float,
    eps: float = LOG_EPS,
    min_log: float = MIN_LOG_PROB
) -> np.ndarray | float:
    """
    Computes numerically safe log-probabilities preventing underflow, -inf, and NaN.
    """
    if isinstance(p, (float, int)):
        if p <= eps:
            return min_log
        return float(np.log(np.clip(p, eps, 1.0)))
    
    p_arr = np.asarray(p, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(p_arr <= eps, min_log, np.log(np.clip(p_arr, eps, 1.0)))
    return result


def safe_log_sum_exp(log_a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable Log-Sum-Exp avoiding intermediate underflow/overflow."""
    max_val = np.max(log_a, axis=axis, keepdims=True)
    max_val_squeezed = np.squeeze(max_val, axis=axis)
    # Mask out -inf / MIN_LOG_PROB
    valid = max_val_squeezed > (MIN_LOG_PROB / 2.0)
    exp_diff = np.exp(np.clip(log_a - max_val, -700.0, 0.0))
    sum_exp = np.sum(exp_diff, axis=axis)
    out = np.where(valid, max_val_squeezed + np.log(np.clip(sum_exp, LOG_EPS, None)), MIN_LOG_PROB)
    return out


# -----------------------------------------------------------------------------
# 5. Temporal Hyperparameter Search Spaces for Optimization
# -----------------------------------------------------------------------------
HYPERPARAMETER_SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    # Stickiness: self-transition probability A[i, i]
    "transition_stickiness": {
        "type": "float",
        "low": 0.88,
        "high": 0.998,
        "default": 0.975,
        "description": "Baseline self-transition diagonal probability"
    },
    # Off-diagonal routing: probability that a possession change routes through Dead Ball
    # In tactical football, >80% of turnovers pass through contest / dead-ball
    "dead_ball_turnover_bias": {
        "type": "float",
        "low": 0.60,
        "high": 0.98,
        "default": 0.85,
        "description": "Probability that leaving H or A routes into Dead Ball rather than direct switch"
    },
    # Bayesian prior temperature: scaling on baseline prior correction
    "prior_temperature": {
        "type": "float",
        "low": 0.3,
        "high": 2.0,
        "default": 1.0,
        "description": "Bayesian inversion temperature scaling: log P(X|S) = log P(S|X) - (1/tau)*log P(S)"
    },
    # Continuous duration penalty weight in HSMM trellis
    "duration_weight": {
        "type": "float",
        "low": 0.1,
        "high": 3.0,
        "default": 1.0,
        "description": "Relative weight of parametric duration log-likelihood in dynamic programming score"
    },
    # Morphological kernel windows (in absolute milliseconds)
    "closing_kernel_ms": {
        "type": "choice",
        "choices": [200.0, 400.0, 600.0, 800.0, 1000.0],
        "default": 600.0,
        "description": "Temporal closing window size in ms (bridges micro-gaps)"
    },
    "opening_kernel_ms": {
        "type": "choice",
        "choices": [0.0, 200.0, 320.0, 480.0, 600.0],
        "default": 320.0,
        "description": "Temporal opening window size in ms (erodes isolated transient flickers)"
    },
    # Parametric Gamma distribution shape k (alpha) and scale theta (beta = 1/theta)
    # Shape k > 1 guarantees p(d -> 0) -> 0, suppressing physically impossible micro-possessions
    "gamma_shape_k": {
        "type": "float",
        "low": 1.5,
        "high": 4.5,
        "default": 2.5,
        "description": "Shape parameter k of continuous duration Gamma distribution"
    },
    # Dynamic context modulation factor for entropy-driven transitions
    "entropy_penalty_factor": {
        "type": "float",
        "low": 0.0,
        "high": 2.0,
        "default": 0.5,
        "description": "Modulates transition matrix inertia based on upstream prediction entropy"
    }
}
