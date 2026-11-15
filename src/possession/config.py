"""Configuration for the FC Barcelona x Metrica Sports possession pipeline.

Architecture
------------
Role 3: estimates P(D) (dead-ball probability).
Role 4: estimates conditional H/A probabilities from the 37 shared SkillCorner
        features. Training is performed only on external SkillCorner matches.
Role 5: combines Role 3, Role 4 and the remaining context/evidence signals from
        the full match Parquet, then decodes H/A/D/X at spell level.

Important contracts
-------------------
* The full match feature-engineering Parquet currently has 111 columns.
* Role 4 uses exactly the 37 columns declared in the SkillCorner V3 manifest.
* The other columns are preserved for downstream context/quality/evidence; they
  are not silently added to Role 4's ML feature matrix.
* P(H)+P(A)=1 at the Role 4 conditional-model output.
* Multiplication by (1-P(D)) is only valid when Role 3's P(D) means probability
  that the ball is out of play, not generic uncertainty/contested possession.
  This semantic handshake must be confirmed before enabling probability fusion.
* X is a decoder abstention decision, not a fourth probability mass.
* No fitting, calibration, threshold tuning or pseudo-labelling is allowed on
  the target match, its video or its ground truth.

Parameters that require empirical selection are deliberately None. Select them
only with external data / external-match OOF predictions or fixed domain rules.
"""

from __future__ import annotations

from copy import deepcopy
from math import ceil
from typing import Any


CONFIG: dict[str, Any] = {
    "project": {
        "name": "FC Barcelona x Metrica Sports - More than a Hack 2027",
        "pipeline_name": "possession",
        "architecture_version": "role3_role4_role5",
        "schema_profile": "full_111",
        "expected_full_parquet_columns": 111,
        "fps_target_match": 25.0,
        "fps_external_training": 10.0,
        "pitch_length_m": 105.3,
        "pitch_width_m": 68.0,
        "team_codes": {"home": 1, "away": -1, "unknown": 0},
        "target_classes": {"home": 0, "away": 1},
        "final_decision_classes": ("H", "A", "D", "X"),
        "live_possession_classes": ("H", "A"),
        "abstain_label": "X",
        "random_state": 42,
    },

    # Time-based constants are canonical. Derive frame counts for the relevant
    # sampling rate instead of reusing 25-FPS frame counts on 10-FPS data.
    "physical_and_temporal": {
        "short_gap_max_s": 1.0,       # formerly 25 frames at 25 FPS
        "contested_radius_m": 2.5,
        "control_radius_m": 3.0,
        "support_radius_m": 5.0,
        "window_short_s": 2.0,
        "window_long_s": 4.0,
        "reaction_time_s": 0.40,
        "player_max_speed_m_s": 7.50,
        "pitch_control_tau_s": 0.50,
        "max_ball_speed_m_s": 38.0,
        "savgol_window_s": 0.28,       # 7 samples at the target 25 FPS
        "savgol_poly": 2,
    },

    # Full target-match Parquet: all engineered/context columns remain available
    # to the later decoder. Do not reduce this table to the 37 Role 4 features.
    "full_match_parquet": {
        "expected_columns": 111,
        # A single-match competition Parquet may not carry match_id. In that
        # case inference treats the file as one match, but a separate Role 3
        # join must still be unambiguous.
        "match_id_column": "match_id",
        "match_id_aliases": ("matchId", "game_id", "gameId", "fixture_id"),
        "allow_single_match_without_match_id": True,
        # In the currently used 111-column competition Parquet the physical
        # frame key is often named `frame`; `frame_id` remains a supported alias.
        "frame_id_column": "frame",
        "frame_id_aliases": ("frame_id", "frame_idx", "frame_index", "frame_number", "frame_num"),
        "allow_row_order_without_frame_id": False,
        "timestamp_column": "timestamp_s",
        "period_column": "period",
        "preserve_all_columns_for_role5": True,
        "allow_unexpected_columns": False,
        "feature_groups": (
            "frame_and_match_context",
            "ball_tracking_motion_and_observability",
            "tactical_camera_and_homography",
            "video_tags_and_set_piece_context",
            "player_visibility",
            "formation_and_team_dynamics",
            "ball_player_proximity_and_pitch_control",
            "possession_history_and_control_windows",
            "goalkeeper_context",
            "broadcast_camera_context",
            "leadlag_transition_features",
        ),
    },

    # Role 4 training contract. Model feature names are sourced from the V3
    # manifest; they are intentionally not duplicated/hard-coded here.
    "role4_training": {
        "enabled": True,
        "training_dataset": "skillcorner_training_v3.parquet",
        "feature_manifest": "skillcorner_feature_manifest_v3.json",
        "conversion_report": "skillcorner_conversion_report_v3.csv",
        "target_column": "target",
        "match_id_column": "match_id",
        "frame_id_column": "frame_id",
        "expected_match_count": 20,
        "expected_model_feature_count": 37,
        "feature_name_source": "skillcorner_feature_manifest_v3.json",
        "feature_selection_policy": "manifest_exact_list_only",
        "allow_automatic_numeric_feature_discovery": False,
        "target_mapping": {"H": 0, "A": 1},
        "label_source_policy": {
            "primary": "phases_of_play.team_in_possession_id",
            "fallback_sources": (
                "tracking.possession.group",
                "dynamic_events.csv",
            ),
            "fallback_usage": "fill_unlabelled_gaps_only",
            "missing_target_policy": "exclude_from_role4_training",
            "conflicting_overlapping_phase_policy": "exclude_ambiguous_frame",
            "manual_labels_allowed": False,
            "target_match_data_allowed": False,
        },
        "fps": 10.0,
        "cross_validation": {
            "strategy": "GroupKFold",
            "n_splits": 5,
            "group_column": "match_id",
            "expected_validation_matches_per_fold": 4,
            "split_by_frames": False,
            "shuffle_frames": False,
            "require_disjoint_match_ids": True,
        },
        "preprocessing": {
            "logistic_regression": {
                "imputer": "median",
                "scaler": "StandardScaler",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 3000,
                "class_weight": "balanced",
            },
            "hist_gradient_boosting": {
                "native_nan_support": True,
                "sample_weight": "balanced_per_training_fold",
                "learning_rate": 0.08,
                "max_iter": 250,
                "max_leaf_nodes": 15,
                "l2_regularization": 1.0,
                "early_stopping": False,
            },
            "max_nan_ratio_per_feature": 1.0,  # report NaN; do not reject solely for NaN
            "reject_infinite_values": True,
        },
        "evaluation": {
            "predictions": "out_of_fold",
            "metrics": (
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_home",
                "f1_away",
                "equal_weight_mean_match_f1_macro",
                "transition_count_ratio",
            ),
            "transition_count_ratio_is_official_spell_f1": False,
            "model_selection_data_scope": "external_skillcorner_oof_only",
            "threshold_tuning_on_target_match": False,
        },
        "inference_outputs": {
            "home_probability_column": "p_home_ha",
            "away_probability_column": "p_away_ha",
            "probabilities_sum_to_one": True,
            "probability_semantics": "conditional_over_the_binary_HA_training_label_space; not automatically calibrated as P(team|ball_in_play)",
        },
        "artifacts_directory": "role4_artifacts",
        "artifacts": (
            "role4_logistic_regression.joblib",
            "role4_hist_gradient_boosting.joblib",
            "role4_training_report.json",
            "role4_oof_predictions.csv",
            "role4_feature_nan_report.csv",
        ),
    },

    # Routing for the 111-column target-match feature set.
    # Role 4 consumes only the 37 manifest-matched inputs. Everything else is
    # retained for Role 5 and should not be silently discarded.
    "feature_routing": {
        "role4_ml_inputs": {
            "source": "role4_training.feature_manifest",
            "expected_count": 37,
            "same_math_definition_required_across_datasets": True,
            "only_manifest_columns_enter_model": True,
        },
        "role5_evidence_groups": (
            "possession_history_and_control_windows",
            "ball_player_proximity_and_pitch_control",
            "formation_and_team_dynamics",
            "goalkeeper_context",
            "leadlag_transition_features",
            "video_tags_and_set_piece_context",
        ),
        "role5_quality_context_groups": (
            "ball_tracking_motion_and_observability",
            "player_visibility",
            "tactical_camera_and_homography",
            "broadcast_camera_context",
        ),
        "role5_general_context_groups": (
            "frame_and_match_context",
            "video_tags_and_set_piece_context",
            "formation_and_team_dynamics",
        ),
        "drop_non_model_columns": False,
        "recompute_tracking_or_homography": False,
    },

    # Role 4 evidence fusion: deterministic, bounded domain rules using
    # competition-only columns. HGB remains unchanged and still consumes only
    # the 37 SkillCorner features. None of the weights below are fitted on the
    # target match. Correlated features are grouped and capped.
    "role4_evidence_fusion": {
        "version": "v2",
        "enabled": True,
        "method": "bounded_log_odds_evidence",
        "max_total_adjustment_log_odds": 0.45,
        "probability_epsilon": 1e-6,
        "uncertainty_gating": True,
        "group_caps_log_odds": {
            "possession_history": 0.25,
            "smart_tagging": 0.12,
            "leadlag": 0.15,
        },
        "rules": (
            {
                "name": "last_controlled_team_weak_prior",
                "column": "last_controlled_team",
                "kind": "team_code",
                "group": "possession_history",
                "home_values": (1, "1", "HOME", "H"),
                "away_values": (-1, "-1", "AWAY", "A"),
                "neutral_values": (0, "0", "UNKNOWN", "NONE", "NAN", "NULL", None),
                "unknown_value_policy": "error",
                "weight_log_odds": 0.20,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                    # frames_since_last_control is measured in target-match frames;
                    # tau=75 frames corresponds to 3 seconds at 25 FPS.
                    {"kind": "exp_decay", "column": "frames_since_last_control", "scale": 75.0},
                ),
                "sign_semantics": "1=Home, -1=Away, 0=neutral; freshness decays over 3 seconds at 25 FPS",
            },
            {
                "name": "consecutive_touches_balance",
                "kind": "paired_count_difference",
                "home_column": "home_consecutive_touches",
                "away_column": "away_consecutive_touches",
                "scale": 3.0,
                "group": "possession_history",
                "weight_log_odds": 0.12,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                    {"kind": "category_map", "column": "ball_status_code", "mapping": {0: 1.0, 1: 0.70, 2: 0.35}},
                    {"kind": "exp_decay", "column": "ball_sigma_m", "scale": 4.0},
                ),
                "sign_semantics": "home_consecutive_touches - away_consecutive_touches; positive=Home",
            },
            {
                "name": "causal_control_diff_2s",
                "column": "control_diff_causal50",
                "kind": "signed_scaled",
                "scale": 3.0,
                "group": "possession_history",
                "weight_log_odds": 0.12,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                    {"kind": "category_map", "column": "ball_status_code", "mapping": {0: 1.0, 1: 0.70, 2: 0.35}},
                    {"kind": "exp_decay", "column": "ball_sigma_m", "scale": 4.0},
                ),
                "sign_semantics": "CONFIRMED: positive=Home advantage; negative=Away advantage",
            },
            {
                "name": "causal_control_diff_4s",
                "column": "control_diff_causal100",
                "kind": "signed_scaled",
                "scale": 4.5,
                "group": "possession_history",
                "weight_log_odds": 0.08,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                    {"kind": "category_map", "column": "ball_status_code", "mapping": {0: 1.0, 1: 0.70, 2: 0.35}},
                    {"kind": "exp_decay", "column": "ball_sigma_m", "scale": 4.0},
                ),
                "sign_semantics": "CONFIRMED: positive=Home advantage; negative=Away advantage",
            },
            {
                "name": "smart_tagging_team_prior",
                "column": "st_team_prior",
                "kind": "team_code",
                "group": "smart_tagging",
                "home_values": (1, "1", "HOME", "H"),
                "away_values": (-1, "-1", "AWAY", "A"),
                "neutral_values": (0, "0", "UNKNOWN", "NONE", "NAN", "NULL", None),
                "unknown_value_policy": "error",
                "weight_log_odds": 0.10,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                    {"kind": "min_ratio", "column": "st_n_active_tags", "scale": 2.0},
                    # Video-tag votes are suppressed when broadcast is invalid,
                    # replay is active, or ball is outside the camera FOV.
                    {"kind": "binary", "column": "brd_valid", "on_value": 1},
                    {"kind": "binary", "column": "brd_is_replay", "on_value": 0},
                    {"kind": "binary", "column": "ball_in_camera_fov", "on_value": 1},
                ),
                "sign_semantics": "1=Home, -1=Away, 0=neutral; tag-count and live-broadcast reliability gates",
            },
            {
                "name": "smart_tagging_team_score",
                "column": "st_team_score",
                "kind": "signed_scaled",
                "scale": 3.0,
                "group": "smart_tagging",
                "weight_log_odds": 0.08,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                    {"kind": "min_ratio", "column": "st_n_active_tags", "scale": 2.0},
                    {"kind": "binary", "column": "brd_valid", "on_value": 1},
                    {"kind": "binary", "column": "brd_is_replay", "on_value": 0},
                    {"kind": "binary", "column": "ball_in_camera_fov", "on_value": 1},
                ),
                "sign_semantics": "CONFIRMED BY FEATURE DESCRIPTION: positive tag-vote difference favors Home; live-broadcast reliability gates",
            },
            {
                "name": "near1_distance_leadlag_04s",
                "column": "near1_diff_leadlag_10f",
                "kind": "signed_scaled",
                "scale": 2.0,
                "group": "leadlag",
                "weight_log_odds": 0.08,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                    {"kind": "category_map", "column": "ball_status_code", "mapping": {0: 1.0, 1: 0.70, 2: 0.35}},
                    {"kind": "exp_decay", "column": "ball_sigma_m", "scale": 4.0},
                    {"kind": "pair_min_ratio", "home_column": "home_n_visible", "away_column": "away_n_visible", "scale": 8.0},
                ),
                "sign_semantics": "ASSUMPTION: positive=Home, inferred from near1_dist_diff_m=(Away distance - Home distance); verify source transform",
            },
            {
                "name": "near1_distance_leadlag_10s",
                "column": "near1_diff_leadlag_25f",
                "kind": "signed_scaled",
                "scale": 3.0,
                "group": "leadlag",
                "weight_log_odds": 0.05,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                    {"kind": "category_map", "column": "ball_status_code", "mapping": {0: 1.0, 1: 0.70, 2: 0.35}},
                    {"kind": "exp_decay", "column": "ball_sigma_m", "scale": 4.0},
                    {"kind": "pair_min_ratio", "home_column": "home_n_visible", "away_column": "away_n_visible", "scale": 8.0},
                ),
                "sign_semantics": "ASSUMPTION: positive=Home, inferred from near1_dist_diff_m=(Away distance - Home distance); verify source transform",
            },
            {
                "name": "pitch_control_leadlag_04s",
                "column": "pitch_ctrl_leadlag_10f",
                "kind": "signed_scaled",
                "scale": 0.25,
                "group": "leadlag",
                "weight_log_odds": 0.08,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                    {"kind": "pair_min_ratio", "home_column": "home_n_visible", "away_column": "away_n_visible", "scale": 8.0},
                ),
                "sign_semantics": "ASSUMPTION: positive change means increasing Home pitch control; verify source transform",
            },
            {
                "name": "pitch_control_leadlag_10s",
                "column": "pitch_ctrl_leadlag_25f",
                "kind": "signed_scaled",
                "scale": 0.35,
                "group": "leadlag",
                "weight_log_odds": 0.05,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                    {"kind": "pair_min_ratio", "home_column": "home_n_visible", "away_column": "away_n_visible", "scale": 8.0},
                ),
                "sign_semantics": "ASSUMPTION: positive change means increasing Home pitch control; verify source transform",
            },
        ),
        # These are deliberately routed as context for Role 5, not H/A votes.
        # A team taking a restart is not necessarily in possession before the
        # ball becomes live; goalkeeper signals already belong to HGB features.
        "context_only_columns": (
            "st_restart_team",
            "st_set_piece",
            "st_throw_in",
            "st_corner_kick",
            "st_goal_kick",
            "st_free_kick",
            "st_kick_off",
            "st_goal",
            "st_post_goal_dead",
            "ball_near_boundary",
            "home_gk_dist_m",
            "away_gk_dist_m",
            "gk_holding_ball_proxy",
            "brd_valid",
            "brd_is_replay",
            "frames_since_brd_replay",
            "homography_reproj_err",
            "ball_in_camera_fov",
            "home_n_visible",
            "away_n_visible",
        ),
        "known_hgb_overlap_not_readded": (
            "home_gk_dist_m",
            "away_gk_dist_m",
            "gk_holding_ball_proxy",
            "home_in_3m",
            "away_in_3m",
            "home_in_5m",
            "away_in_5m",
            "home_min_tti_s",
            "away_min_tti_s",
            "tti_diff_s",
            "home_pitch_control",
        ),
        "strict_unknown_categories": True,
        "missing_evidence_policy": "neutral_no_vote",
        "weights_are_learned": False,
        "target_match_tuning_allowed": False,
        "note": (
            "Fixed and bounded domain heuristics only. control_diff_causal50/100 positive=Home is confirmed by the project owner. "
            "Positive sign for lead-lag features is inferred from their base feature definitions and must be checked against the feature code. "
            "Goalkeeper columns already present among the 37 HGB inputs are not voted again. Restart-team is context-only, not live possession evidence."
        ),
    },

    # Role 3 produces P(D). Semantics must be agreed between Role 3 and Role 5.
    "role3_output": {
        "probability_column": "p_dead",
        "expected_range": (0.0, 1.0),
        # Set to 'out_of_play_only' only after Role 3 confirms this exact meaning.
        # Do not use the fusion formula if this includes contested in-play frames.
        "confirmed_semantics": None,
        "required_semantics_for_fusion": "out_of_play_only",
    },

    # Probability-level combination. Kept disabled until the Role 3 semantic
    # handshake is confirmed. No learned calibration is fitted on the target.
    "probability_fusion": {
        "enabled": False,
        "method": "conditional_live_mass_scaling",
        "assumption": "Role4 H/A probabilities must be a defensible approximation of team possession conditional on live play; Role3 must output P(ball_out_of_play). This is not guaranteed merely by binary H/A training.",
        "formulas": {
            "p_home": "(1 - p_dead) * p_home_ha",
            "p_away": "(1 - p_dead) * p_away_ha",
            "p_dead": "p_dead",
        },
        "require_role3_semantics_confirmation": True,
        "role4_conditional_semantics_confirmed": False,
        "require_role4_semantics_confirmation": True,
        "contested_in_play_is_dead_ball": False,
        "sum_to_one_tolerance": 1e-9,
        "require_finite_probabilities": True,
        "clip_input_probabilities": False,  # out-of-range input should raise, not be silently fixed
    },

    # Role 5 is the spell-level decoder. Thresholds are deliberately unset:
    # they must be selected using external games / domain rules only.
    "role5_decoder": {
        "enabled": False,  # enable after Role 4 and Role 3 contracts are approved
        "architecture": "spell_aware_temporal_state_machine",
        "output_classes": ("H", "A", "D", "X"),
        "abstain_label": "X",
        "abstain_has_probability_mass": False,
        "components": (
            "evidence_accumulation",
            "hysteresis",
            "temporal_state_machine",
            "spell_boundary_handling",
        ),
        "inputs": (
            "p_home",
            "p_away",
            "p_dead",
            "role5_evidence_groups",
            "role5_quality_context_groups",
            "role5_general_context_groups",
        ),
        "temporal_order": {
            "match_column": "match_id",
            "frame_column": "frame_id",
            "fps_target_match": 25.0,
            "reset_state_at_period_boundary": True,
            "do_not_carry_state_between_matches": True,
        },
        "decision_parameters": {
            "home_enter_threshold": None,
            "away_enter_threshold": None,
            "switch_hysteresis_margin": None,
            "evidence_accumulation_tau_s": None,
            "minimum_spell_duration_s": None,
            "abstain_uncertainty_threshold": None,
            "dead_ball_enter_threshold": None,
            "dead_ball_exit_threshold": None,
        },
        "quality_policy": {
            "low_quality_reduces_evidence_weight": True,
            "missing_context_does_not_imply_zero_evidence": True,
            "weights_and_cutoffs": None,
        },
        "calibration_scope": "external_matches_only_or_fixed_domain_rules",
        "target_match_tuning_allowed": False,
        "pseudo_labels_from_target_match_allowed": False,
    },

    "export_contract": {
        "class_order_probability_mass": ("H", "A", "D"),
        "decision_label_column": "possession_label",
        "home_probability_column": "p_home",
        "away_probability_column": "p_away",
        "dead_ball_probability_column": "p_dead",
        "match_id_column": "match_id",
        "frame_id_column": "frame_id",
        "output_format": "parquet",
        "preserve_original_111_columns": True,
        "preserve_frame_order": True,
        "require_finite_probabilities": True,
        "require_probability_sum_one_for_had": True,
        "probability_sum_tolerance": 1e-9,
        "x_is_decision_not_probability_class": True,
    },

    "competition_compliance": {
        "training_data_scope": "external_datasets_with_usage_rights_only",
        "target_video_training_or_tuning_allowed": False,
        "target_ground_truth_training_or_tuning_allowed": False,
        "manual_target_match_labels_allowed": False,
        "target_match_pseudo_labels_allowed": False,
        "target_match_output_checked_against_video_allowed": False,
        "target_match_used_for_probability_calibration": False,
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
        elif node is None:
            found.append(path)

    walk(cfg, "")
    return found


# Stable contracts are checked on import. Data-dependent/semantic selections are
# checked only when the corresponding downstream stage is enabled.
validate_project_config()
validate_role4_evidence_config()
validate_pipeline_structure()
validate_probability_fusion_config()


if __name__ == "__main__":
    print("CONFIG: OK")
    print("Full-match columns:", CONFIG["full_match_parquet"]["expected_columns"])
    print("Role 4 model features:", CONFIG["role4_training"]["expected_model_feature_count"])
    print("Role 4 temporal sampling:", get_temporal_sampling_config(CONFIG["role4_training"]["fps"]))
    print("Target temporal sampling:", get_temporal_sampling_config(CONFIG["project"]["fps_target_match"]))
    pending = unresolved_parameters()
    print("Unresolved parameters:", len(pending))
    for item in pending:
        print(" -", item)
