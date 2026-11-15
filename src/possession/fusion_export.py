"""Fusion & Export: P(H|Live), P(A|Live) + P(D) -> H/A/D for Role 5.

Stage 6. Implements only the agreed formula and validation:

    P(H) = (1 - P(D)) * P(H | Live)
    P(A) = (1 - P(D)) * P(A | Live)
    P(D) = P(D)

It does NOT implement Role 5's Viterbi decoder and does not alter P(D).
The file contract with Role 5 (columns, dtypes, format) is PROVISIONAL and
driven by CONFIG["fusion_export"] until it is agreed.

Strictness: frame_id sets must match exactly. Order is taken from the Student
output; Role 3's table is re-ordered by frame_id explicitly (never positionally
assumed). Missing, extra or duplicated ids, and NaN/out-of-range P(D), raise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config import CONFIG
from contracts import (
    ContractError,
    DeadBallProbabilities,
    HADProbabilities,
    LiveProbabilities,
)


class ExportConfigError(ValueError):
    """Export configuration is missing or invalid."""


def align_dead_ball(live: LiveProbabilities, dead: DeadBallProbabilities) -> np.ndarray:
    """Return P(D) re-ordered to ``live.frame_id`` order; ids must match exactly."""
    live_ids = pd.Index(live.frame_id)
    dead_ids = pd.Index(dead.frame_id)
    if len(live_ids) != len(dead_ids) or set(live_ids) != set(dead_ids):
        missing = len(set(live_ids) - set(dead_ids))
        extra = len(set(dead_ids) - set(live_ids))
        raise ContractError(f"frame_id mismatch between Student and P(D): missing={missing}, extra={extra}")
    return pd.Series(dead.p_dead, index=dead_ids).reindex(live_ids).to_numpy()


def fuse_had(
    live: LiveProbabilities,
    dead: DeadBallProbabilities,
    config: Mapping[str, Any] | None = None,
) -> HADProbabilities:
    """Combine conditional live probabilities with P(D) into (H, A, D)."""
    cfg = (CONFIG if config is None else config)["fusion_export"]
    p_dead = align_dead_ball(live, dead)
    p_h = (1.0 - p_dead) * live.probs[:, 0]
    p_a = (1.0 - p_dead) * live.probs[:, 1]
    result = HADProbabilities(
        frame_id=live.frame_id,
        probs=np.column_stack([p_h, p_a, p_dead]),
        sum_tolerance=float(cfg["probability_sum_tolerance"]),
    )
    return result


def validate_had(had: HADProbabilities, config: Mapping[str, Any] | None = None) -> None:
    """Re-validate a final table (e.g. after loading) against the config contract."""
    cfg = (CONFIG if config is None else config)["fusion_export"]
    HADProbabilities(
        frame_id=had.frame_id,
        probs=had.probs,
        sum_tolerance=float(cfg["probability_sum_tolerance"]),
    )


def to_export_frame(had: HADProbabilities, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Build the export table with config-defined column names and frame order."""
    cfg = (CONFIG if config is None else config)["fusion_export"]
    names = [cfg["home_output_column"], cfg["away_output_column"], cfg["dead_ball_output_column"]]
    if len({cfg["frame_id_column"], *names}) != 4:
        raise ExportConfigError("Export column names must be distinct (incl. frame_id column).")
    validate_had(had, config)
    table = pd.DataFrame(had.probs, columns=names)
    table.insert(0, cfg["frame_id_column"], had.frame_id)
    return table


def export_had(
    had: HADProbabilities,
    path: str | Path,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Write the export table in ``output_format`` ("parquet" or "csv")."""
    cfg = (CONFIG if config is None else config)["fusion_export"]
    table = to_export_frame(had, config)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = cfg["output_format"]
    if fmt == "parquet":
        try:
            table.to_parquet(out, index=False)
        except ImportError as exc:
            raise ImportError("Parquet export needs pyarrow in the uv environment.") from exc
    elif fmt == "csv":
        table.to_csv(out, index=False)
    else:
        raise ExportConfigError(f"Unsupported output_format {fmt!r}")
    return out
