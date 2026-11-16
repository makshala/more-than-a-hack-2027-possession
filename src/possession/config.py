"""Configuration for the FC Barcelona x Metrica Sports possession pipeline.

Architecture
------------
Role 3: estimates P(D) (dead-ball probability).
Role 4: estimates conditional H/A probabilities from the 37 shared SkillCorner
        features. Training is performed only on external SkillCorner matches.
Role 5: combines Role 3, Role 4 and the remaining context/evidence signals from
        the full match Parquet, then decodes H/A/D/X at spell level.
"""

from __future__ import annotations

from copy import deepcopy
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

    "physical_and_temporal": {
        "short_gap_max_s": 1.0,
        "contested_radius_m": 2.5,
        "control_radius_m": 3.0,
        "support_radius_m": 5.0,
        "window_short_s": 2.0,
        "window_long_s": 4.0,
        "reaction_time_s": 0.40,
        "player_max_speed_m_s": 7.50,
        "pitch_control_tau_s": 0.50,
        "max_ball_speed_m_s": 38.0,
        "savgol_window_s": 0.28,
        "savgol_poly": 2,
    },

    "full_match_parquet": {
        "expected_columns": 111,
        "match_id_column": "match_id",
        "match_id_aliases": ("matchId", "game_id", "gameId", "fixture_id"),
        "allow_single_match_without_match_id": True,
        "frame_id_column": "frame",
        "frame_id_aliases": (
            "frame_id", "frame_idx", "frame_index", "frame_number", "frame_num"
        ),
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
            "primary": "tracking.possession.group",
            "fallback_sources": (
                "dynamic_events.csv",
                "phases_of_play.team_in_possession_id",
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
            "max_nan_ratio_per_feature": 1.0,
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
            "probability_semantics": (
                "conditional_over_the_binary_HA_training_label_space; "
                "not automatically calibrated as P(team|ball_in_play)"
            ),
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

    "role4_evidence_fusion": {
        "version": "v4_spell_responsive",
        "enabled": True,
        "method": "bounded_log_odds_evidence",
        "model_logit_temperature": 1.85,
        "max_total_adjustment_log_odds": 1.80,
        "probability_epsilon": 1e-6,
        "uncertainty_gating": False,
        "group_caps_log_odds": {
            "direct_ball_contact": 1.45,
            "possession_history": 0.75,
            "smart_tagging": 0.40,
            "leadlag": 0.05,
        },
           "rules": (
            {
                "name": "direct_near1_distance_advantage",
                "column": "near1_dist_diff_m",
                "kind": "signed_scaled",
                "scale": 1.4,
                "group": "direct_ball_contact",
                "weight_log_odds": 0.90,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                    {
                        "kind": "category_map",
                        "column": "ball_status_code",
                        "mapping": {0: 1.0, 1: 0.95, 2: 0.90}, 
                    },
                ),
                "sign_semantics": "away_near1_dist_m - home_near1_dist_m; positive means Home player is closer to the ball",
            },
            {
                "name": "players_in_3m_control_zone",
                "kind": "paired_count_difference",
                "home_column": "home_in_3m",
                "away_column": "away_in_3m",
                "scale": 1.0,
                "group": "direct_ball_contact",
                "weight_log_odds": 0.65,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                    {
                        "kind": "category_map",
                        "column": "ball_status_code",
                        "mapping": {0: 1.0, 1: 0.95, 2: 0.90}, 
                    },
                ),
                "sign_semantics": "home_in_3m - away_in_3m; overrides formation bias during short 2-3s turnovers",
            },
            {
                "name": "tti_advantage_instant",
                "column": "tti_diff_s",
                "kind": "signed_scaled",
                "scale": 0.55,
                "group": "direct_ball_contact",
                "weight_log_odds": 0.40,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                ),
                "sign_semantics": "away_min_tti_s - home_min_tti_s; positive means Home reaches ball faster",
            },
            {
                "name": "centered_control_diff_2s",
                "column": "control_diff_roll50",
                "kind": "signed_scaled",
                "scale": 2.0,
                "group": "possession_history",
                "weight_log_odds": 0.50,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                ),
                "sign_semantics": "Zero-lag centered 50-frame control window",
            },
            {
                "name": "consecutive_touches_balance",
                "kind": "paired_count_difference",
                "home_column": "home_consecutive_touches",
                "away_column": "away_consecutive_touches",
                "scale": 1.5,
                "group": "possession_history",
                "weight_log_odds": 0.35,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                ),
                "sign_semantics": "home_consecutive_touches - away_consecutive_touches",
            },
            {
                "name": "smart_tagging_restart_team",
                "column": "st_restart_team",
                "kind": "team_code",
                "group": "smart_tagging",
                "home_values": (1, "1", "HOME", "H"),
                "away_values": (-1, "-1", "AWAY", "A"),
                "neutral_values": (0, "0", "UNKNOWN", "NONE", "NAN", "NULL", None),
                "unknown_value_policy": "error",
                "weight_log_odds": 0.30,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                ),
                "sign_semantics": "1=Home restart, -1=Away restart",
            },
            {
                "name": "smart_tagging_team_score",
                "column": "st_team_score",
                "kind": "signed_scaled",
                "scale": 2.5,
                "group": "smart_tagging",
                "weight_log_odds": 0.15,
                "quality_gates": (
                    {"kind": "binary", "column": "is_graded", "on_value": 1},
                ),
                "sign_semantics": "Positive tag-vote difference favors Home",
            },
           
        ),
        "context_only_columns": (
            "st_set_piece",
            "st_throw_in",
            "st_corner_kick",
            "st_goal_kick",
            "st_free_kick",
            "st_kick_off",
            "st_goal",
            "st_post_goal_dead",
            "ball_near_boundary",
            "brd_valid",
            "brd_is_replay",
            "frames_since_brd_replay",
            "homography_reproj_err",
            "ball_in_camera_fov",
            "home_n_visible",
            "away_n_visible",
        ),
        "known_hgb_overlap_not_readded": (),
        "strict_unknown_categories": True,
        "missing_evidence_policy": "neutral_no_vote",
        "weights_are_learned": False,
        "target_match_tuning_allowed": False,
    },

    "role3_output": {
        "probability_column": "p_dead",
        "expected_range": (0.0, 1.0),
        "confirmed_semantics": None,
        "required_semantics_for_fusion": "out_of_play_only",
    },

    "probability_fusion": {
        "enabled": False,
        "method": "conditional_live_mass_scaling",
        "assumption": (
            "Role4 H/A probabilities must be a defensible approximation of team possession conditional on live play; "
            "Role3 must output P(ball_out_of_play). This is not guaranteed merely by binary H/A training."
        ),
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
        "clip_input_probabilities": False,
    },

    "role5_decoder": {
        "enabled": False,
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
    """Return an independent copy of the configuration."""
    return deepcopy(CONFIG)


def frames_for_seconds(seconds: float, fps: float) -> int:
    """Convert a time window to a sample count for the supplied FPS."""
    if seconds < 0:
        raise ValueError("seconds must be >= 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    return int(round(seconds * fps))


def get_temporal_sampling_config(fps: float) -> dict[str, Any]:
    """Return time-derived frame counts without mixing target and training FPS."""
    temporal = CONFIG["physical_and_temporal"]
    savgol_frames = frames_for_seconds(temporal["savgol_window_s"], fps)
    if savgol_frames % 2 == 0:
        savgol_frames += 1
    savgol_frames = max(savgol_frames, int(temporal["savgol_poly"]) + 1)
    if savgol_frames % 2 == 0:
        savgol_frames += 1
    return {
        "fps": float(fps),
        "dt_s": 1.0 / float(fps),
        "short_gap_max_frames": frames_for_seconds(temporal["short_gap_max_s"], fps),
        "window_short_frames": frames_for_seconds(temporal["window_short_s"], fps),
        "window_long_frames": frames_for_seconds(temporal["window_long_s"], fps),
        "reaction_time_frames": frames_for_seconds(temporal["reaction_time_s"], fps),
        "savgol_window_frames": savgol_frames,
    }


def validate_project_config(config: dict[str, Any] | None = None) -> None:
    """Validate stable architecture and data contracts."""
    cfg = CONFIG if config is None else config
    project = cfg["project"]
    if project["schema_profile"] != "full_111":
        raise ValueError("schema_profile must be 'full_111'.")
    if project["expected_full_parquet_columns"] != 111:
        raise ValueError("The full feature-engineering Parquet contract expects 111 columns.")
    full_match = cfg["full_match_parquet"]
    if full_match.get("expected_columns") != 111:
        raise ValueError("full_match_parquet.expected_columns must be 111.")
    if tuple(project["final_decision_classes"]) != ("H", "A", "D", "X"):
        raise ValueError("Final decision classes must be exactly ('H', 'A', 'D', 'X').")
    if project["target_classes"] != {"home": 0, "away": 1}:
        raise ValueError("Role 4 target mapping must be Home=0, Away=1.")

    role4 = cfg["role4_training"]
    if role4["expected_model_feature_count"] != 37:
        raise ValueError("Role 4 must consume exactly 37 manifest-declared features.")
    if role4["cross_validation"]["strategy"] != "GroupKFold":
        raise ValueError("Role 4 validation must use GroupKFold by match_id.")
    if role4["cross_validation"]["split_by_frames"]:
        raise ValueError("Frame-wise random splitting is forbidden.")
    if role4["cross_validation"]["group_column"] != role4["match_id_column"]:
        raise ValueError("GroupKFold group column must equal the match_id column.")

    if cfg["feature_routing"]["drop_non_model_columns"]:
        raise ValueError("Non-model columns must be preserved for Role 5.")
    if cfg["competition_compliance"]["target_video_training_or_tuning_allowed"]:
        raise ValueError("Training/tuning on the target video is prohibited.")
    if cfg["competition_compliance"]["target_ground_truth_training_or_tuning_allowed"]:
        raise ValueError("Training/tuning on target ground truth is prohibited.")


def validate_probability_fusion_config(config: dict[str, Any] | None = None) -> None:
    """Validate the semantic precondition for P(D)-weighted H/A fusion."""
    cfg = CONFIG if config is None else config
    role3_semantics = cfg["role3_output"].get("confirmed_semantics")
    required = cfg["role3_output"].get("required_semantics_for_fusion")
    fusion = cfg["probability_fusion"]
    if fusion["enabled"] and role3_semantics != required:
        raise ValueError(
            "Probability fusion is enabled, but Role 3 has not confirmed that "
            "P(D) means out-of-play probability only. Do not treat contested "
            "in-play frames as dead ball."
        )
    if fusion["enabled"] and fusion.get("require_role4_semantics_confirmation", True):
        if not fusion.get("role4_conditional_semantics_confirmed", False):
            raise ValueError(
                "Probability fusion is enabled, but the Role 4 H/A probabilities "
                "have not been validated as conditional live-play possession "
                "probabilities. Binary H/A training alone does not prove this."
            )


def validate_role4_evidence_config(config: dict[str, Any] | None = None) -> None:
    """Validate explicit Role 4 evidence rules without fitting any weights."""
    cfg = CONFIG if config is None else config
    block = cfg["role4_evidence_fusion"]
    if block.get("method") != "bounded_log_odds_evidence":
        raise ValueError("Role 4 evidence method must be bounded_log_odds_evidence.")
    if float(block.get("max_total_adjustment_log_odds", -1)) < 0:
        raise ValueError("Role 4 evidence global log-odds cap must be >= 0.")
    if block.get("enabled", False) and not block.get("rules"):
        raise ValueError("Role 4 evidence fusion is enabled but no evidence rules are configured.")
    seen_names: set[str] = set()
    allowed_kinds = {
        "team_code",
        "signed",
        "signed_scaled",
        "home_probability",
        "paired_count_difference",
    }
    for rule in block.get("rules", []):
        name = rule.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValueError("Every Role 4 evidence rule must have a unique non-empty name.")
        seen_names.add(name)
        kind = rule.get("kind")
        if kind not in allowed_kinds:
            raise ValueError(f"Unsupported evidence kind for {name!r}: {kind!r}.")
        if kind == "paired_count_difference":
            if not rule.get("home_column") or not rule.get("away_column"):
                raise ValueError(f"Paired evidence rule {name!r} requires home_column and away_column.")
        elif not isinstance(rule.get("column"), str) or not rule.get("column"):
            raise ValueError(f"Evidence rule {name!r} must define a column.")
        weight = float(rule.get("weight_log_odds", -1))
        if weight < 0:
            raise ValueError(f"Evidence weight for {name!r} must be >= 0.")
        if kind in {"signed_scaled", "paired_count_difference"} and float(rule.get("scale", 0)) <= 0:
            raise ValueError(f"Evidence scale for {name!r} must be > 0.")
    if block.get("target_match_tuning_allowed", True):
        raise ValueError("Role 4 evidence weights must never be tuned on the target match.")


def validate_pipeline_structure(config: dict[str, Any] | None = None) -> None:
    """Check that the current Role 3/4/5 sections exist."""
    cfg = CONFIG if config is None else config
    required_sections = (
        "project",
        "physical_and_temporal",
        "full_match_parquet",
        "role4_training",
        "feature_routing",
        "role4_evidence_fusion",
        "role3_output",
        "probability_fusion",
        "role5_decoder",
        "export_contract",
        "competition_compliance",
    )
    missing = [key for key in required_sections if not isinstance(cfg.get(key), dict)]
    if missing:
        raise ValueError(f"Missing or invalid config sections: {missing}")


def unresolved_parameters(config: dict[str, Any] | None = None) -> list[str]:
    """Return explicitly unset parameters; informational only."""
    cfg = CONFIG if config is None else config
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif value is None:
            found.append(path)

    walk(cfg, "")
    return found


validate_project_config()
validate_role4_evidence_config()
validate_pipeline_structure()
validate_probability_fusion_config()


if __name__ == "__main__":
    print("CONFIG: OK")