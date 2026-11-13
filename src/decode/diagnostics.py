"""
Temporal diagnostics, Temporal Intersection over Union (tIoU), SpellF1 calculation,
and aggressive fragmentation penalty analysis for tactical possession estimation.
Frame-level accuracy is deprecated in favor of temporal segment overlap and boundary precision.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from config import (
    STATE_MAP, INV_STATE_MAP, STATE_HOME, STATE_AWAY, STATE_DEAD_BALL,
    frames_to_ms, DEFAULT_FPS
)


class TemporalDiagnostics:
    """
    Evaluates possession sequence decoding quality using:
    1. Temporal Intersection over Union (tIoU) bipartite matching.
    2. SpellF1 metric at multiple tIoU thresholds.
    3. Aggressive fragmentation penalty targeting false micro-spell splits.
    4. Boundary jitter analysis in milliseconds.
    """

    @staticmethod
    def calculate_temporal_iou(
        pred_start: int, pred_end: int,
        gt_start: int, gt_end: int
    ) -> float:
        """Computes exact temporal 1D Intersection over Union between two intervals."""
        intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start) + 1)
        union = max(pred_end, gt_end) - min(pred_start, gt_start) + 1
        if union <= 0:
            return 0.0
        return float(intersection / union)

    @classmethod
    def evaluate_spells(
        cls,
        df_pred_spells: pd.DataFrame,
        df_gt_spells: pd.DataFrame,
        tiou_threshold: float = 0.5,
        fragmentation_decay: float = 0.75,
        fps: float = DEFAULT_FPS
    ) -> Dict[str, Any]:
        """
        Computes Spell-level Precision, Recall, SpellF1 at given tIoU threshold,
        and applies an aggressive geometric fragmentation penalty.
        """
        if df_gt_spells.empty:
            return {
                "spell_f1_penalized": 0.0,
                "spell_f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "fragmentation_penalty": 1.0,
                "mean_tiou": 0.0
            }

        if df_pred_spells.empty:
            return {
                "spell_f1_penalized": 0.0,
                "spell_f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "fragmentation_penalty": 0.0,
                "mean_tiou": 0.0
            }

        total_tp = 0
        total_fp = 0
        total_fn = 0
        matched_tious: List[float] = []
        boundary_jitters_frames: List[float] = []

        class_metrics: Dict[str, Dict[str, float]] = {}

        # Evaluate per class (H, A, D)
        unique_classes = set(df_gt_spells["state"].unique()).union(set(df_pred_spells["state"].unique()))

        # For fragmentation analysis: count how many predicted spells overlap each GT spell
        gt_fragments: Dict[int, int] = {gt_id: 0 for gt_id in df_gt_spells["spell_id"]}

        for cls_name in ["H", "A", "D"]:
            gt_cls = df_gt_spells[df_gt_spells["state"] == cls_name]
            pred_cls = df_pred_spells[df_pred_spells["state"] == cls_name]

            num_gt = len(gt_cls)
            num_pred = len(pred_cls)

            if num_gt == 0 and num_pred == 0:
                continue

            if num_gt == 0:
                total_fp += num_pred
                class_metrics[cls_name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": num_pred, "fn": 0}
                continue

            if num_pred == 0:
                total_fn += num_gt
                class_metrics[cls_name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": num_gt}
                continue

            # Build pairwise tIoU cost matrix
            cost_matrix = np.zeros((num_gt, num_pred), dtype=np.float64)
            gt_records = gt_cls.to_dict("records")
            pred_records = pred_cls.to_dict("records")

            for i, gt_row in enumerate(gt_records):
                for j, pred_row in enumerate(pred_records):
                    iou = cls.calculate_temporal_iou(
                        pred_row["start_frame"], pred_row["end_frame"],
                        gt_row["start_frame"], gt_row["end_frame"]
                    )
                    cost_matrix[i, j] = iou

                    # Check for fragmentation overlap: if pred overlaps GT substantially (>15% overlap or >5 frames)
                    overlap = max(0, min(pred_row["end_frame"], gt_row["end_frame"]) - max(pred_row["start_frame"], gt_row["start_frame"]) + 1)
                    if overlap >= 5:
                        gt_fragments[gt_row["spell_id"]] += 1

            # Solve maximum weight bipartite matching via Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(-cost_matrix)

            cls_tp = 0
            cls_tious = []

            for r, c in zip(row_ind, col_ind):
                iou = cost_matrix[r, c]
                if iou >= tiou_threshold:
                    cls_tp += 1
                    cls_tious.append(iou)
                    matched_tious.append(iou)

                    # Compute boundary jitter
                    gt_row = gt_records[r]
                    pred_row = pred_records[c]
                    jitter = (abs(pred_row["start_frame"] - gt_row["start_frame"]) +
                              abs(pred_row["end_frame"] - gt_row["end_frame"])) / 2.0
                    boundary_jitters_frames.append(jitter)

            cls_fp = num_pred - cls_tp
            cls_fn = num_gt - cls_tp

            total_tp += cls_tp
            total_fp += cls_fp
            total_fn += cls_fn

            prec = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0.0
            rec = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0.0
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            class_metrics[cls_name] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "tp": cls_tp,
                "fp": cls_fp,
                "fn": cls_fn,
                "mean_tiou": float(np.mean(cls_tious)) if cls_tious else 0.0
            }

        # Overall Precision, Recall, SpellF1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        spell_f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Aggressive Geometric Fragmentation Penalty:
        # If a single ground truth spell is falsely split into K > 1 micro-spells by the decoder,
        # it receives an exponential/geometric penalty factor: decay^(K - 1).
        excess_fragments = sum(max(0, count - 1) for count in gt_fragments.values())
        fragmented_gt_count = sum(1 for count in gt_fragments.values() if count > 1)
        
        fragmentation_penalty = float(fragmentation_decay ** excess_fragments)
        spell_f1_penalized = float(spell_f1 * fragmentation_penalty)

        mean_jitter_ms = float(frames_to_ms(np.mean(boundary_jitters_frames), fps)) if boundary_jitters_frames else 0.0

        return {
            "spell_f1_penalized": spell_f1_penalized,
            "spell_f1": float(spell_f1),
            "precision": float(precision),
            "recall": float(recall),
            "fragmentation_penalty": fragmentation_penalty,
            "excess_fragments": excess_fragments,
            "fragmented_gt_spells": fragmented_gt_count,
            "mean_tiou": float(np.mean(matched_tious)) if matched_tious else 0.0,
            "mean_boundary_jitter_ms": mean_jitter_ms,
            "class_metrics": class_metrics
        }

    @classmethod
    def compute_diagnostics(
        cls,
        df_source: pd.DataFrame,
        decoded_states: np.ndarray,
        df_spells: pd.DataFrame,
        fps: float = DEFAULT_FPS
    ) -> Dict[str, Any]:
        """
        Generates comprehensive temporal diagnostic report deprecating naive frame-accuracy
        in favor of tIoU, SpellF1, and fragmentation penalties.
        """
        total_frames = len(decoded_states)
        distribution = {
            name: float(np.mean(decoded_states == s_id))
            for s_id, name in STATE_MAP.items()
        }

        # Spell transition statistics
        spell_seq = df_spells["state"].to_numpy() if not df_spells.empty else np.array([])
        transitions = {}
        for src, dst in zip(spell_seq[:-1], spell_seq[1:]):
            key = f"{src} -> {dst}"
            transitions[key] = transitions.get(key, 0) + 1

        # Compute active tactical possession breakdown (excluding Dead Ball)
        total_home_frames = int(np.sum(decoded_states == STATE_HOME))
        total_away_frames = int(np.sum(decoded_states == STATE_AWAY))
        total_dead_frames = int(np.sum(decoded_states == STATE_DEAD_BALL))
        in_play_frames = total_home_frames + total_away_frames

        if in_play_frames > 0:
            tactical_possession_pct = {
                "Home": float(np.around((total_home_frames / in_play_frames) * 100.0, decimals=2)),
                "Away": float(np.around((total_away_frames / in_play_frames) * 100.0, decimals=2)),
                "In_Play_Ratio_Pct": float(np.around((in_play_frames / total_frames) * 100.0, decimals=2)),
                "Dead_Ball_Ratio_Pct": float(np.around((total_dead_frames / total_frames) * 100.0, decimals=2))
            }
        else:
            tactical_possession_pct = {"Home": 0.0, "Away": 0.0, "In_Play_Ratio_Pct": 0.0, "Dead_Ball_Ratio_Pct": 100.0}

        # Argmax smoothing / divergence rate
        dead_col = "p_dead_ball" if "p_dead_ball" in df_source.columns else "p_dead" if "p_dead" in df_source.columns else None
        argmax_divergence = None
        if dead_col and all(c in df_source.columns for c in ["p_home", "p_away", dead_col]):
            raw_argmax = np.argmax(df_source[["p_home", "p_away", dead_col]].to_numpy(), axis=1)
            argmax_divergence = float(np.mean(raw_argmax != decoded_states))

        # Per-state duration breakdown
        state_duration_stats = {}
        for s_name in ["H", "A", "D"]:
            s_spells = df_spells[df_spells["state"] == s_name]
            if not s_spells.empty:
                state_duration_stats[s_name] = {
                    "count": len(s_spells),
                    "total_seconds": float(np.around(s_spells["duration_seconds"].sum(), decimals=2)),
                    "mean_seconds": float(np.around(s_spells["duration_seconds"].mean(), decimals=2)),
                    "median_seconds": float(np.around(s_spells["duration_seconds"].median(), decimals=2)),
                    "min_seconds": float(np.around(s_spells["duration_seconds"].min(), decimals=2)),
                    "max_seconds": float(np.around(s_spells["duration_seconds"].max(), decimals=2))
                }
            else:
                state_duration_stats[s_name] = {"count": 0, "total_seconds": 0.0, "mean_seconds": 0.0, "median_seconds": 0.0, "min_seconds": 0.0, "max_seconds": 0.0}

        # Spells per half if available
        half_breakdown = {}
        if "half" in df_spells.columns:
            for half_name, grp in df_spells.groupby("half"):
                h_home = int(grp[grp["state"] == "H"]["duration_frames"].sum())
                h_away = int(grp[grp["state"] == "A"]["duration_frames"].sum())
                h_inplay = h_home + h_away
                half_breakdown[str(half_name)] = {
                    "spells_count": len(grp),
                    "home_pct": float(np.around((h_home / h_inplay) * 100.0, decimals=1)) if h_inplay > 0 else 0.0,
                    "away_pct": float(np.around((h_away / h_inplay) * 100.0, decimals=1)) if h_inplay > 0 else 0.0,
                    "duration_seconds": float(np.around(grp["duration_seconds"].sum(), decimals=2))
                }

        # Ground truth evaluation
        spell_metrics = None
        frame_acc_deprecated = None

        if "ground_truth" in df_source.columns:
            gt_states = df_source["ground_truth"].to_numpy()
            frame_acc_deprecated = float(np.mean(gt_states == decoded_states))

            from spells import PossessionSpellExtractor
            df_gt_spells = PossessionSpellExtractor.extract_spells(
                df_source, gt_states, apply_morphology=False, apply_hysteresis=False, fps=fps
            )

            # Evaluate at standard tIoU=0.5
            spell_metrics = cls.evaluate_spells(
                df_pred_spells=df_spells,
                df_gt_spells=df_gt_spells,
                tiou_threshold=0.5,
                fps=fps
            )

        return {
            "total_frames": total_frames,
            "total_spells": len(df_spells),
            "total_duration_seconds": float(np.around(total_frames / fps, decimals=2)),
            "tactical_possession_pct": tactical_possession_pct,
            "state_distribution": distribution,
            "state_duration_stats": state_duration_stats,
            "half_breakdown": half_breakdown,
            "argmax_divergence": argmax_divergence,
            "spell_metrics": spell_metrics,
            "frame_level_accuracy_DEPRECATED": frame_acc_deprecated,
            "mean_spell_duration_seconds": float(df_spells["duration_seconds"].mean()) if not df_spells.empty else 0.0,
            "min_spell_duration_seconds": float(df_spells["duration_seconds"].min()) if not df_spells.empty else 0.0,
            "spell_transitions": transitions
        }
