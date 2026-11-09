"""Local configuration for the possession pipeline.

This file is the single source of truth for experiment settings in the
`possession` project. It is intentionally independent of any repository-wide
configuration. `uv` manages the Python environment and dependencies; this file
manages pipeline behaviour and experiment parameters.

The active schema profile is the 111-column version, including Broadcast and
lead-lag features, matching the Feature Engineering version already run by the
project owner.

Parameters that require empirical inspection are left as `None` deliberately.
The relevant stage should fail with a clear message until those values are
selected; no data-dependent thresholds are guessed here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CONFIG: dict[str, Any] = {
    "project": {
        "name": "possession",
        "schema_profile": "full_111",
        "expected_input_columns": 111,
        "class_order_had": ("H", "A"),
        "class_order_had_full": ("H", "A", "D"),
        "team_codes": {"home": 1, "away": -1, "unknown": 0},
        "fps": 25,
    },

    "feature_engineering": {
        "schema_profile": "full_111",
        "include_broadcast": True,
        "include_lead_lag": True,
        "add_signed_features": False,
        "include_gk_distance_contrast": False,
        "allow_extra_columns": False,
        "identifier_columns": ("frame_id",),
        "context_columns": (
            "period",
            "is_graded",
            "is_pre_kickoff_d",
            "timestamp_s",
        ),
    },

    "quality_gating": {
        "ball": {
            # Relative quality assignments are left for explicit calibration.
            "status_quality": {
                "OBSERVED": None,
                "INTERPOLATED_SHORT": None,
                "INFERRED_LATENT": None,
            },
            "fov_quality": {0: None, 1: None},
            "sigma_scale_m": None,
            "gap_scale_frames": None,
            # None means equal component weights in the current implementation.
            "component_weights": None,
        },
        "players": {
            "expected_players_per_team": 11,
            "component_weights": None,
        },
        "homography": {
            "reprojection_error_scale": None,
        },
        "history": {
            # This currently controls freshness decay. We will separate
            # observation quality from historical relevance before finalizing.
            "age_decay_frames": None,
            "component_weights": None,
        },
        "formation": {
            "component_weights": None,
        },
        "pitch": {
            "component_weights": None,
        },
        "broadcast": {
            "replay_recovery_frames": None,
            "component_weights": None,
        },
        "diagnostics": {
            "report_quantiles": (0.10, 0.25, 0.50, 0.75, 0.90, 0.99),
            "check_temporal_stability": True,
            "check_monotonicity": True,
            "run_sensitivity_analysis": True,
        },
    },

    "labeling_functions": {
        "class_order": ("H", "A"),
        "abstain_label": "ABSTAIN",
        "experts": (
            "possession_history",
            "geometry",
            "video_tags",
            "goalkeeper",
            "formation",
        ),
        # None = undecided; stages that need it fail with a clear message.
        # "causal_only" forbids NON_CAUSAL_FEATURES (feature_engineering.py);
        # "allow_non_causal" permits offline look-ahead features.
        "causality_policy": None,
        # Which q_* columns gate each expert. PROVISIONAL proposal: confirm
        # after local diagnostics. A vote is replaced by ABSTAIN when ANY listed
        # q_* is below that expert's cutoff.
        "quality_columns": {
            "possession_history": ("q_history",),
            "geometry": ("q_ball", "q_players", "q_homography", "q_pitch"),
            "video_tags": ("q_tags",),
            "goalkeeper": ("q_ball", "q_players"),
            "formation": ("q_formation", "q_players"),
        },
        # Per-expert quality cutoffs will be set after local diagnostics.
        # Missing / None => the LF refuses to run (no default is invented).
        "quality_cutoffs": {},
        # Expert-specific parameters (thresholds, windows ...). Each LF class
        # lists in `required_params` what it needs; none are guessed here.
        "params": {
            "possession_history": {},
            "geometry": {},
            "video_tags": {},
            "goalkeeper": {},
            "formation": {},
        },
        "emit_debug_columns": True,
    },

    "label_model": {
        # "snorkel": adapter skeleton (optional dependency, see label_model.py)
        # "majority_vote": plain reference baseline, NOT a recommended model.
        "backend": "snorkel",
        "class_order": ("H", "A"),
        "cardinality": 2,
        # Hierarchy of correlated LFs: {group_name: (lf_name, ...)}.
        # Defined after LF coverage/agreement/correlation analysis.
        "dependency_groups": {},
        "fit_on_time_blocks": True,
        # Backend hyperparameters (e.g. epochs, lr, seed). Unset on purpose.
        "snorkel": {
            "n_epochs": None,
            "lr": None,
            "l2": None,
            "seed": None,
        },
        # Rows excluded from fitting (e.g. known dead-ball frames). None = undecided.
        "training_row_filter": None,
    },

    "student_model": {
        "models_to_compare": ("logistic", "gam"),
        "target_type": "soft_labels",
        "class_order": ("H", "A"),
        "random_state": None,
        "features": {
            # Names from FEATURE_GROUPS, or "all_predictors". None = undecided.
            "groups": None,
            "include_columns": (),
            "exclude_columns": (),
            # Identifier/context columns are never predictors by default.
            "exclude_identifier_and_context": True,
            # Same vocabulary as labeling_functions.causality_policy.
            "causality_policy": None,
            # "error": fail if any selected predictor is NaN. Other policies
            # (imputation, missing-indicator, native NaN support) are TODO and
            # must be chosen deliberately: ~41% of ball information is uncertain.
            "missing_value_policy": None,
        },
        "temporal_validation": {
            "strategy": "blocked",
            "purge_frames": None,
            "n_blocks": None,
        },
        "ablation": {
            # "leave_one_group_out" | None (undecided)
            "scheme": None,
        },
        "logistic": {
            # sklearn >= 1.8: penalty is deprecated; C + l1_ratio define it
            # (l1_ratio=0 -> L2, 1 -> L1, between -> elastic net).
            "C": None,
            "l1_ratio": None,
            "standardize": None,
            "solver": None,
            "max_iter": None,
            # None is a legitimate value here (= no class weighting), so it is
            # not reported as unresolved.
            "class_weight": None,
        },
        "gam": {
            "library": None,
            "n_splines": None,
            "smoothing": None,
            "interactions": (),
        },
    },

    "fusion_export": {
        "class_order": ("H", "A", "D"),
        "conditional_live_classes": ("H", "A"),
        "dead_ball_probability_source": "role_3",
        "home_output_column": "p_home",
        "away_output_column": "p_away",
        "dead_ball_output_column": "p_dead_ball",
        "frame_id_column": "frame_id",
        "require_finite_probabilities": True,
        "require_probability_sum_one": True,
        # Floating-point tolerance for the sum check (numerical, not modelling).
        "probability_sum_tolerance": 1e-9,
        "preserve_frame_order": True,
        # "parquet" (target) or "csv" (local debugging). Final contract with
        # Role 5 is still to be agreed.
        "output_format": "parquet",
    },
}


def get_config() -> dict[str, Any]:
    """Return an independent copy so experiments cannot mutate CONFIG globally."""
    return deepcopy(CONFIG)


def validate_project_config(config: dict[str, Any] | None = None) -> None:
    """Validate fixed project-level contracts without guessing tuning values."""
    cfg = CONFIG if config is None else config
    project = cfg.get("project", {})

    if project.get("schema_profile") != "full_111":
        raise ValueError("This project configuration expects schema_profile='full_111'.")
    if project.get("expected_input_columns") != 111:
        raise ValueError("The active Feature Engineering schema must contain 111 columns.")
    if tuple(project.get("class_order_had", ())) != ("H", "A"):
        raise ValueError("The H/A class order must be exactly ('H', 'A').")
    if tuple(project.get("class_order_had_full", ())) != ("H", "A", "D"):
        raise ValueError("The final class order must be exactly ('H', 'A', 'D').")

    feature_config = cfg.get("feature_engineering", {})
    if not feature_config.get("include_broadcast") or not feature_config.get("include_lead_lag"):
        raise ValueError(
            "The active schema profile is full_111 and must include Broadcast and lead-lag features."
        )


def validate_quality_gating_config(config: dict[str, Any] | None = None) -> None:
    """Fail early with a readable list of quality parameters still to choose.

    This is intentionally separate from project-contract validation: the
    project can be configured and imported before empirical quality scales
    have been selected.
    """
    cfg = CONFIG if config is None else config
    quality = cfg.get("quality_gating", {})
    required_paths = (
        ("ball", "status_quality", "OBSERVED"),
        ("ball", "status_quality", "INTERPOLATED_SHORT"),
        ("ball", "status_quality", "INFERRED_LATENT"),
        ("ball", "fov_quality", 0),
        ("ball", "fov_quality", 1),
        ("ball", "sigma_scale_m"),
        ("ball", "gap_scale_frames"),
        ("homography", "reprojection_error_scale"),
        ("history", "age_decay_frames"),
        ("broadcast", "replay_recovery_frames"),
    )
    missing: list[str] = []
    for path in required_paths:
        value: Any = quality
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is None:
            missing.append("quality_gating." + ".".join(map(str, path)))

    if missing:
        raise ValueError(
            "Quality Gating parameters still require local selection: "
            + ", ".join(missing)
        )


# Validate invariant settings at import time. Data-dependent tuning parameters
# are intentionally not validated until the Quality Gating stage is run.
validate_project_config()


# Keys that must exist (values may be None) so every stage can read its section.
_REQUIRED_STRUCTURE: dict[str, tuple[str, ...]] = {
    "labeling_functions": ("experts", "causality_policy", "quality_columns",
                           "quality_cutoffs", "params", "emit_debug_columns"),
    "label_model": ("backend", "class_order", "dependency_groups", "snorkel",
                    "fit_on_time_blocks"),
    "student_model": ("models_to_compare", "features", "temporal_validation",
                      "ablation", "logistic", "gam"),
    "fusion_export": ("class_order", "home_output_column", "away_output_column",
                      "dead_ball_output_column", "frame_id_column",
                      "probability_sum_tolerance", "output_format"),
}


def validate_pipeline_structure(config: dict[str, Any] | None = None) -> None:
    """Check that every downstream section/key exists (values may be None)."""
    cfg = CONFIG if config is None else config
    problems: list[str] = []
    for section, keys in _REQUIRED_STRUCTURE.items():
        block = cfg.get(section)
        if not isinstance(block, dict):
            problems.append(f"missing section {section!r}")
            continue
        problems += [f"missing key {section}.{k}" for k in keys if k not in block]
    if problems:
        raise ValueError("CONFIG structure is incomplete: " + "; ".join(problems))
    experts = tuple(cfg["labeling_functions"]["experts"])
    for name in ("quality_columns", "params"):
        extra = set(cfg["labeling_functions"][name]) - set(experts)
        if extra:
            raise ValueError(f"labeling_functions.{name} has unknown experts: {sorted(extra)}")


def unresolved_parameters(config: dict[str, Any] | None = None) -> list[str]:
    """List dotted paths of parameters still set to None (the TODO register).

    Informational only: it never fails. `student_model.logistic.class_weight`
    is skipped because None is a valid, meaningful value there.
    """
    cfg = CONFIG if config is None else config
    skip = {"student_model.logistic.class_weight", "student_model.random_state"}
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif node is None and path not in skip:
            found.append(path)

    walk(cfg, "")
    return found


validate_pipeline_structure()
