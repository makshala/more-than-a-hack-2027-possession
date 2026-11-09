# Quality Gating — local diagnostic report

> This report is descriptive. Candidate scales are not approved thresholds, and q values are heuristic quality indices, not calibrated probabilities.

## Dataset scope

- Rows before filtering: **199,319**
- Rows after filtering: **158,857**
- Filters: is_graded == 1, period in (1, 2)
- Available columns: **111**
- Missing expected quality-input columns: none

## Numeric input profile (selected first rows)

| column | dtype | n_rows | n_non_null_original | n_finite_numeric | missing_pct | non_numeric_pct_of_non_null | infinite_count | negative_pct_of_finite | zero_pct_of_finite | min | max | mean | std | p10 | p25 | p50 | p75 | p90 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ball_status_code | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 64.4 | 0 | 2 | 0.7114 | 0.9571 | 0 | 0 | 0 | 2 | 2 | 2 |
| ball_sigma_m | float32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0 | 0.5 | 12.07 | 2.065 | 2.139 | 0.5 | 0.5 | 0.5 | 4.296 | 5.351 | 6.81 |
| ball_gap_frames | int32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 64.4 | 0 | 3062 | 297.9 | 579.4 | 0 | 0 | 0 | 394 | 1141 | 3062 |
| frames_since_ball_gap | int32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 35.54 | 0 | 1e+04 | 657.6 | 1176 | 0 | 0 | 223 | 915 | 1804 | 3971 |
| frames_to_ball_gap | int32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 35.54 | 0 | 4089 | 576.6 | 785.9 | 0 | 0 | 220 | 893 | 1723 | 3317 |
| ball_in_camera_fov | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0.4778 | 0 | 1 | 0.9952 | 0.06896 | 1 | 1 | 1 | 1 | 1 | 1 |
| ball_near_boundary | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 96.01 | 0 | 1 | 0.03994 | 0.1958 | 0 | 0 | 0 | 0 | 0 | 1 |
| homography_reproj_err | float32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0 | 0.06228 | 5.453 | 0.2964 | 0.5119 | 0.1748 | 0.2 | 0.2308 | 0.2645 | 0.3078 | 4.295 |
| camera_pan_speed | float32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0.8618 | 0 | 1105 | 30.1 | 47.42 | 0.5516 | 1.246 | 4.709 | 42.46 | 96.92 | 203.2 |
| home_n_visible | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0.4702 | 0 | 19 | 10.19 | 1.972 | 9 | 10 | 11 | 11 | 11 | 12 |
| away_n_visible | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0 | 1 | 15 | 10.27 | 1.738 | 9 | 10 | 11 | 11 | 11 | 12 |
| last_controlled_team | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 47.44 | 0.09631 | -1 | 1 | 0.05031 | 0.9983 | -1 | -1 | 1 | 1 | 1 | 1 |
| frames_since_last_control | int32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 27.98 | 0 | 1e+04 | 262.4 | 707.3 | 0 | 0 | 29 | 213 | 647 | 4097 |
| home_pitch_control | float32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0.4318 | 0 | 1 | 0.4929 | 0.2485 | 0.158 | 0.323 | 0.4868 | 0.6551 | 0.8554 | 0.998 |
| home_mean_speed | float32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0.4702 | 0 | 8.004 | 1.683 | 0.9891 | 0.6264 | 0.9634 | 1.461 | 2.221 | 3.043 | 4.771 |
| away_mean_speed | float32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0 | 0.00875 | 7.827 | 1.681 | 1 | 0.6087 | 0.9442 | 1.449 | 2.245 | 3.104 | 4.758 |
| home_length_x | float32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0.9852 | 0 | 87.45 | 31.07 | 10.96 | 17.5 | 23.86 | 30.61 | 38.57 | 46.04 | 53.49 |
| away_length_x | float32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0.6314 | 0 | 89.02 | 28.72 | 11.13 | 15.24 | 22.01 | 27.99 | 34.99 | 44.03 | 54.27 |
| home_width_y | float32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0.9852 | 0 | 53.61 | 24.26 | 7.858 | 15.41 | 19.36 | 23.68 | 29.73 | 35.12 | 40.21 |
| away_width_y | float32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0.6314 | 0 | 52.14 | 22.73 | 6.688 | 14.55 | 18.35 | 22.94 | 26.57 | 31.26 | 37.62 |
| all_players_mean_speed | float32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0 | 0.0434 | 6.296 | 1.687 | 0.9504 | 0.6771 | 0.9756 | 1.463 | 2.229 | 2.975 | 4.686 |
| players_walking_ratio | float32 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 0.4123 | 0 | 1 | 0.584 | 0.29 | 0.15 | 0.3333 | 0.6111 | 0.8571 | 0.9524 | 1 |
| players_in_penalty_box | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 1.368 | 0 | 17 | 3.17 | 3.629 | 1 | 1 | 1 | 4 | 10 | 14 |
| st_set_piece | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 72.33 | 0 | 1 | 0.2767 | 0.4473 | 0 | 0 | 0 | 1 | 1 | 1 |
| st_throw_in | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 86.05 | 0 | 1 | 0.1395 | 0.3464 | 0 | 0 | 0 | 0 | 1 | 1 |
| st_corner_kick | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 93.38 | 0 | 1 | 0.06617 | 0.2486 | 0 | 0 | 0 | 0 | 0 | 1 |
| st_goal_kick | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 94.49 | 0 | 1 | 0.05514 | 0.2283 | 0 | 0 | 0 | 0 | 0 | 1 |
| st_free_kick | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 97.24 | 0 | 1 | 0.02757 | 0.1637 | 0 | 0 | 0 | 0 | 0 | 1 |
| st_kick_off | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 99.58 | 0 | 1 | 0.004205 | 0.06471 | 0 | 0 | 0 | 0 | 0 | 0 |
| st_goal | int8 | 158857 | 158857 | 158857 | 0 | 0 | 0 | 0 | 89.61 | 0 | 1 | 0.1039 | 0.3051 | 0 | 0 | 0 | 0 | 1 | 1 |

## Basic domain checks

| column | expected_rule | n_checked | n_invalid | invalid_pct_of_checked |
| --- | --- | --- | --- | --- |
| is_graded | must be 0 or 1 | 158857 | 0 | 0 |
| is_pre_kickoff_d | must be 0 or 1 | 158857 | 0 | 0 |
| ball_in_camera_fov | must be 0 or 1 | 158857 | 0 | 0 |
| ball_near_boundary | must be 0 or 1 | 158857 | 0 | 0 |
| brd_valid | must be 0 or 1 | 158857 | 0 | 0 |
| brd_is_replay | must be 0 or 1 | 158857 | 0 | 0 |
| st_set_piece | must be 0 or 1 | 158857 | 0 | 0 |
| st_throw_in | must be 0 or 1 | 158857 | 0 | 0 |
| st_corner_kick | must be 0 or 1 | 158857 | 0 | 0 |
| st_goal_kick | must be 0 or 1 | 158857 | 0 | 0 |
| st_free_kick | must be 0 or 1 | 158857 | 0 | 0 |
| st_kick_off | must be 0 or 1 | 158857 | 0 | 0 |
| st_goal | must be 0 or 1 | 158857 | 0 | 0 |
| st_post_goal_dead | must be 0 or 1 | 158857 | 0 | 0 |
| st_defending_tag | must be 0 or 1 | 158857 | 0 | 0 |
| st_in_box_tag | must be 0 or 1 | 158857 | 0 | 0 |
| is_contested | must be 0 or 1 | 158857 | 0 | 0 |
| last_controlled_team | must be one of -1, 0, 1 | 158857 | 0 | 0 |
| st_restart_team | must be one of -1, 0, 1 | 158857 | 0 | 0 |
| st_team_prior | must be one of -1, 0, 1 | 158857 | 0 | 0 |
| gk_holding_ball_proxy | must be one of -1, 0, 1 | 158857 | 0 | 0 |
| ball_sigma_m | must be >= 0 | 158857 | 0 | 0 |
| ball_gap_frames | must be >= 0 | 158857 | 0 | 0 |
| frames_since_ball_gap | must be >= 0 | 158857 | 0 | 0 |
| frames_to_ball_gap | must be >= 0 | 158857 | 0 | 0 |
| homography_reproj_err | must be >= 0 | 158857 | 0 | 0 |
| home_n_visible | must be >= 0 | 158857 | 0 | 0 |
| away_n_visible | must be >= 0 | 158857 | 0 | 0 |
| home_min_tti_s | must be >= 0 | 158857 | 0 | 0 |
| away_min_tti_s | must be >= 0 | 158857 | 0 | 0 |
| home_near1_dist_m | must be >= 0 | 158857 | 0 | 0 |
| away_near1_dist_m | must be >= 0 | 158857 | 0 | 0 |
| frames_since_last_control | must be >= 0 | 158857 | 0 | 0 |
| frames_since_brd_replay | must be >= 0 | 158857 | 0 | 0 |
| brd_lookat_speed_ms | must be >= 0 | 158857 | 0 | 0 |
| ball_to_brd_lookat_dist_m | must be >= 0 | 158857 | 0 | 0 |
| players_walking_ratio | must be in [0.0, 1.0] | 158857 | 0 | 0 |
| home_pitch_control | must be in [0.0, 1.0] | 158857 | 0 | 0 |
| period | expected coding: 0, 1, or 2 | 158857 | 0 | 0 |
| ball_status | expected one of ['INFERRED_LATENT', 'INTERPOLATED_SHORT', 'OBSERVED'] | 158857 | 0 | 0 |
| home_n_visible | expected integer count in [0, 11] | 158857 | 12836 | 8.08 |
| away_n_visible | expected integer count in [0, 11] | 158857 | 7354 | 4.629 |

## Candidate scales

The candidate scale is the selected positive-value quantile of its source feature. For a decay transform, q(scale) = exp(-1) ≈ 0.368. For replay recovery, q(scale) = 1 - exp(-1) ≈ 0.632.
`age_decay_frames` describes the freshness/relevance of the last-control history. It is not a measure of whether the historical observation itself was recorded correctly; keep that conceptual distinction when finalizing Quality Gating.

| parameter | source_column | score_transform | candidate_quantile | candidate_scale | n_eligible_rows | n_valid_nonnegative | n_positive_for_scale | zero_pct_of_all_rows | quality_at_candidate_scale | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sigma_scale_m | ball_sigma_m | decay | 0.5 | 0.5 | 158857 | 158857 | 158857 | 0 | 0.3679 | candidate only; requires domain review |
| sigma_scale_m | ball_sigma_m | decay | 0.75 | 4.296 | 158857 | 158857 | 158857 | 0 | 0.3679 | candidate only; requires domain review |
| sigma_scale_m | ball_sigma_m | decay | 0.9 | 5.351 | 158857 | 158857 | 158857 | 0 | 0.3679 | candidate only; requires domain review |
| gap_scale_frames | ball_gap_frames | decay | 0.5 | 603 | 158857 | 158857 | 56549 | 64.4 | 0.3679 | candidate only; requires domain review |
| gap_scale_frames | ball_gap_frames | decay | 0.75 | 1207 | 158857 | 158857 | 56549 | 64.4 | 0.3679 | candidate only; requires domain review |
| gap_scale_frames | ball_gap_frames | decay | 0.9 | 1659 | 158857 | 158857 | 56549 | 64.4 | 0.3679 | candidate only; requires domain review |
| reprojection_error_scale | homography_reproj_err | decay | 0.5 | 0.2308 | 158857 | 158857 | 158857 | 0 | 0.3679 | candidate only; requires domain review |
| reprojection_error_scale | homography_reproj_err | decay | 0.75 | 0.2645 | 158857 | 158857 | 158857 | 0 | 0.3679 | candidate only; requires domain review |
| reprojection_error_scale | homography_reproj_err | decay | 0.9 | 0.3078 | 158857 | 158857 | 158857 | 0 | 0.3679 | candidate only; requires domain review |
| age_decay_frames | frames_since_last_control | decay | 0.5 | 93 | 158704 | 158704 | 114261 | 27.98 | 0.3679 | candidate only; requires domain review |
| age_decay_frames | frames_since_last_control | decay | 0.75 | 346 | 158704 | 158704 | 114261 | 27.98 | 0.3679 | candidate only; requires domain review |
| age_decay_frames | frames_since_last_control | decay | 0.9 | 875 | 158704 | 158704 | 114261 | 27.98 | 0.3679 | candidate only; requires domain review |
| replay_recovery_frames | frames_since_brd_replay | recovery | 0.5 | 665 | 106145 | 106145 | 106145 | 33.18 | 0.6321 | candidate only; requires domain review |
| replay_recovery_frames | frames_since_brd_replay | recovery | 0.75 | 1332 | 106145 | 106145 | 106145 | 33.18 | 0.6321 | candidate only; requires domain review |
| replay_recovery_frames | frames_since_brd_replay | recovery | 0.9 | 2170 | 106145 | 106145 | 106145 | 33.18 | 0.6321 | candidate only; requires domain review |

## Sensitivity summary

| parameter | candidate_quantile | candidate_scale | quality_median | quality_p10 | quality_p90 | pct_quality_below_low_threshold | pct_quality_above_high_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sigma_scale_m | 0.5 | 0.5 | 0.3679 | 2.248e-05 | 0.3679 | 40.68 | 0 |
| sigma_scale_m | 0.75 | 4.296 | 0.8901 | 0.2877 | 0.8901 | 0.8479 | 59.33 |
| sigma_scale_m | 0.9 | 5.351 | 0.9108 | 0.3679 | 0.9108 | 0.1429 | 64.44 |
| gap_scale_frames | 0.5 | 603 | 1 | 0.1507 | 1 | 10.48 | 65.13 |
| gap_scale_frames | 0.75 | 1207 | 1 | 0.3886 | 1 | 1.928 | 70.74 |
| gap_scale_frames | 0.9 | 1659 | 1 | 0.5027 | 1 | 1.928 | 73.59 |
| reprojection_error_scale | 0.5 | 0.2308 | 0.3679 | 0.2636 | 0.469 | 4.277 | 0 |
| reprojection_error_scale | 0.75 | 0.2645 | 0.4178 | 0.3123 | 0.5165 | 2.505 | 0 |
| reprojection_error_scale | 0.9 | 0.3078 | 0.4724 | 0.3679 | 0.5668 | 1.704 | 0.0006295 |
| age_decay_frames | 0.5 | 93 | 0.7321 | 0.000952 | 1 | 29.87 | 45.32 |
| age_decay_frames | 0.75 | 346 | 0.9196 | 0.1541 | 1 | 11.66 | 61.84 |
| age_decay_frames | 0.9 | 875 | 0.9674 | 0.4774 | 1 | 3.829 | 73.78 |
| replay_recovery_frames | 0.5 | 665 | 0.3407 | 0 | 0.9313 | 43.04 | 22.17 |
| replay_recovery_frames | 0.75 | 1332 | 0.1878 | 0 | 0.7374 | 51.04 | 6.864 |
| replay_recovery_frames | 0.9 | 2170 | 0.1198 | 0 | 0.5599 | 59.78 | 2.108 |

## Per-period candidate scales

| period | parameter | source_column | score_transform | candidate_quantile | candidate_scale | n_eligible_rows | n_valid_nonnegative | n_positive_for_scale | zero_pct_of_all_rows | quality_at_candidate_scale | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sigma_scale_m | ball_sigma_m | decay | 0.5 | 0.5 | 79467 | 79467 | 79467 | 0 | 0.3679 | candidate only; requires domain review |
| 1 | sigma_scale_m | ball_sigma_m | decay | 0.75 | 4.183 | 79467 | 79467 | 79467 | 0 | 0.3679 | candidate only; requires domain review |
| 1 | sigma_scale_m | ball_sigma_m | decay | 0.9 | 5.25 | 79467 | 79467 | 79467 | 0 | 0.3679 | candidate only; requires domain review |
| 1 | gap_scale_frames | ball_gap_frames | decay | 0.5 | 722 | 79467 | 79467 | 28358 | 64.31 | 0.3679 | candidate only; requires domain review |
| 1 | gap_scale_frames | ball_gap_frames | decay | 0.75 | 1258 | 79467 | 79467 | 28358 | 64.31 | 0.3679 | candidate only; requires domain review |
| 1 | gap_scale_frames | ball_gap_frames | decay | 0.9 | 1659 | 79467 | 79467 | 28358 | 64.31 | 0.3679 | candidate only; requires domain review |
| 1 | reprojection_error_scale | homography_reproj_err | decay | 0.5 | 0.2237 | 79467 | 79467 | 79467 | 0 | 0.3679 | candidate only; requires domain review |
| 1 | reprojection_error_scale | homography_reproj_err | decay | 0.75 | 0.2582 | 79467 | 79467 | 79467 | 0 | 0.3679 | candidate only; requires domain review |
| 1 | reprojection_error_scale | homography_reproj_err | decay | 0.9 | 0.3039 | 79467 | 79467 | 79467 | 0 | 0.3679 | candidate only; requires domain review |
| 1 | age_decay_frames | frames_since_last_control | decay | 0.5 | 101 | 79314 | 79314 | 56392 | 28.84 | 0.3679 | candidate only; requires domain review |
| 1 | age_decay_frames | frames_since_last_control | decay | 0.75 | 371 | 79314 | 79314 | 56392 | 28.84 | 0.3679 | candidate only; requires domain review |
| 1 | age_decay_frames | frames_since_last_control | decay | 0.9 | 843 | 79314 | 79314 | 56392 | 28.84 | 0.3679 | candidate only; requires domain review |
| 1 | replay_recovery_frames | frames_since_brd_replay | recovery | 0.5 | 731 | 53407 | 53407 | 53407 | 32.79 | 0.6321 | candidate only; requires domain review |
| 1 | replay_recovery_frames | frames_since_brd_replay | recovery | 0.75 | 1516 | 53407 | 53407 | 53407 | 32.79 | 0.6321 | candidate only; requires domain review |
| 1 | replay_recovery_frames | frames_since_brd_replay | recovery | 0.9 | 2765 | 53407 | 53407 | 53407 | 32.79 | 0.6321 | candidate only; requires domain review |
| 2 | sigma_scale_m | ball_sigma_m | decay | 0.5 | 0.5 | 79390 | 79390 | 79390 | 0 | 0.3679 | candidate only; requires domain review |
| 2 | sigma_scale_m | ball_sigma_m | decay | 0.75 | 4.44 | 79390 | 79390 | 79390 | 0 | 0.3679 | candidate only; requires domain review |
| 2 | sigma_scale_m | ball_sigma_m | decay | 0.9 | 5.432 | 79390 | 79390 | 79390 | 0 | 0.3679 | candidate only; requires domain review |
| 2 | gap_scale_frames | ball_gap_frames | decay | 0.5 | 577 | 79390 | 79390 | 28191 | 64.49 | 0.3679 | candidate only; requires domain review |
| 2 | gap_scale_frames | ball_gap_frames | decay | 0.75 | 1141 | 79390 | 79390 | 28191 | 64.49 | 0.3679 | candidate only; requires domain review |
| 2 | gap_scale_frames | ball_gap_frames | decay | 0.9 | 3062 | 79390 | 79390 | 28191 | 64.49 | 0.3679 | candidate only; requires domain review |
| 2 | reprojection_error_scale | homography_reproj_err | decay | 0.5 | 0.2369 | 79390 | 79390 | 79390 | 0 | 0.3679 | candidate only; requires domain review |
| 2 | reprojection_error_scale | homography_reproj_err | decay | 0.75 | 0.2699 | 79390 | 79390 | 79390 | 0 | 0.3679 | candidate only; requires domain review |
| 2 | reprojection_error_scale | homography_reproj_err | decay | 0.9 | 0.3113 | 79390 | 79390 | 79390 | 0 | 0.3679 | candidate only; requires domain review |
| 2 | age_decay_frames | frames_since_last_control | decay | 0.5 | 85 | 79390 | 79390 | 57869 | 27.11 | 0.3679 | candidate only; requires domain review |
| 2 | age_decay_frames | frames_since_last_control | decay | 0.75 | 321 | 79390 | 79390 | 57869 | 27.11 | 0.3679 | candidate only; requires domain review |
| 2 | age_decay_frames | frames_since_last_control | decay | 0.9 | 917 | 79390 | 79390 | 57869 | 27.11 | 0.3679 | candidate only; requires domain review |
| 2 | replay_recovery_frames | frames_since_brd_replay | recovery | 0.5 | 609 | 52738 | 52738 | 52738 | 33.57 | 0.6321 | candidate only; requires domain review |
| 2 | replay_recovery_frames | frames_since_brd_replay | recovery | 0.75 | 1191 | 52738 | 52738 | 52738 | 33.57 | 0.6321 | candidate only; requires domain review |
| 2 | replay_recovery_frames | frames_since_brd_replay | recovery | 0.9 | 1725 | 52738 | 52738 | 52738 | 33.57 | 0.6321 | candidate only; requires domain review |

## Monotonicity checks

| parameter | source_column | candidate_quantile | candidate_scale | expected_direction | spearman_rho_input_vs_q | direction_ok | status | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sigma_scale_m | ball_sigma_m | 0.5 | 0.5 | negative | -1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| sigma_scale_m | ball_sigma_m | 0.75 | 4.296 | negative | -1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| sigma_scale_m | ball_sigma_m | 0.9 | 5.351 | negative | -1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| gap_scale_frames | ball_gap_frames | 0.5 | 603 | negative | -1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| gap_scale_frames | ball_gap_frames | 0.75 | 1207 | negative | -1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| gap_scale_frames | ball_gap_frames | 0.9 | 1659 | negative | -1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| reprojection_error_scale | homography_reproj_err | 0.5 | 0.2308 | negative | -1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| reprojection_error_scale | homography_reproj_err | 0.75 | 0.2645 | negative | -1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| reprojection_error_scale | homography_reproj_err | 0.9 | 0.3078 | negative | -1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| age_decay_frames | frames_since_last_control | 0.5 | 93 | negative | -1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| age_decay_frames | frames_since_last_control | 0.75 | 346 | negative | -1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| age_decay_frames | frames_since_last_control | 0.9 | 875 | negative | -1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| replay_recovery_frames | frames_since_brd_replay | 0.5 | 665 | positive | 1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| replay_recovery_frames | frames_since_brd_replay | 0.75 | 1332 | positive | 1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |
| replay_recovery_frames | frames_since_brd_replay | 0.9 | 2170 | positive | 1 | True | pass | Formula is monotonic by construction; this checks the generated component against the local data and index alignment. |

## High absolute Spearman correlations

| feature_a | feature_b | spearman_rho | abs_rho | n_pairwise |
| --- | --- | --- | --- | --- |
| brd_valid | brd_is_replay | -1 | 1 | 158857 |
| control_diff_roll50 | control_diff_roll100 | 0.9744 | 0.9744 | 158857 |
| near1_dist_diff_m | tti_diff_s | 0.972 | 0.972 | 158857 |
| ball_status_code | ball_gap_frames | 0.9689 | 0.9689 | 158857 |
| away_min_tti_s | away_near1_dist_m | 0.9658 | 0.9658 | 158857 |
| home_min_tti_s | home_near1_dist_m | 0.9633 | 0.9633 | 158857 |
| control_diff_causal50 | control_diff_causal100 | 0.9445 | 0.9445 | 158857 |
| control_diff_roll100 | control_diff_causal50 | 0.9404 | 0.9404 | 158857 |
| tti_diff_s | home_pitch_control | 0.9375 | 0.9375 | 158857 |
| ball_status_code | ball_sigma_m | 0.9324 | 0.9324 | 158857 |
| control_diff_roll50 | control_diff_causal50 | 0.9222 | 0.9222 | 158857 |
| near1_dist_diff_m | home_pitch_control | 0.9151 | 0.9151 | 158857 |
| ball_sigma_m | ball_gap_frames | 0.9121 | 0.9121 | 158857 |

## Temporal behaviour of candidate quality components

| parameter | candidate_quantile | candidate_scale | low_quality_pct | low_quality_run_median_frames | low_quality_run_p95_frames | p95_abs_frame_delta | pct_frame_deltas_above_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sigma_scale_m | 0.5 | 0.5 | 40.68 | 262.5 | 1264 | 2.35e-05 | 0.1781 |
| sigma_scale_m | 0.75 | 4.296 | 0.8479 | 28.5 | 160.6 | 0.003609 | 0.1643 |
| sigma_scale_m | 0.9 | 5.351 | 0.1429 | 9 | 70 | 0.003521 | 0.1442 |
| gap_scale_frames | 0.5 | 603 | 10.48 | 1278 | 2500 | 0 | 0.1467 |
| gap_scale_frames | 0.75 | 1207 | 1.928 | 3062 | 3062 | 0 | 0.08813 |
| gap_scale_frames | 0.9 | 1659 | 1.928 | 3062 | 3062 | 0 | 0.06925 |
| reprojection_error_scale | 0.5 | 0.2308 | 4.277 | 4 | 50.4 | 0.03045 | 0 |
| reprojection_error_scale | 0.75 | 0.2645 | 2.505 | 5 | 41.6 | 0.03028 | 0.001259 |
| reprojection_error_scale | 0.9 | 0.3078 | 1.704 | 5 | 53 | 0.02957 | 0.001259 |
| age_decay_frames | 0.5 | 93 | 29.87 | 162 | 1165 | 0.01036 | 0.6131 |
| age_decay_frames | 0.75 | 346 | 11.66 | 221 | 1356 | 0.002861 | 0.1498 |
| age_decay_frames | 0.9 | 875 | 3.829 | 246.5 | 3368 | 0.001138 | 0.07239 |
| replay_recovery_frames | 0.5 | 665 | 43.04 | 513 | 1945 | 0.001355 | 0.06106 |
| replay_recovery_frames | 0.75 | 1332 | 51.04 | 756 | 2823 | 0.0007126 | 0.05162 |
| replay_recovery_frames | 0.9 | 2170 | 59.78 | 981 | 3741 | 0.0004463 | 0.04218 |

## Plots

Plots were not generated (matplotlib unavailable or no plottable columns).

## Interpretation checklist

1. Check that negative values and non-finite values are either impossible by definition or explained.
2. Review each scale candidate in its physical unit (metres or frames), not only its resulting q distribution.
3. Compare the three candidate quantiles and look for excessively collapsed scores near 0 or 1.
4. Inspect long low-quality runs and large frame-to-frame changes; decide whether these match actual observation failures.
5. Do not select a scale because it creates a preferred possession distribution. Quality Gating must not decide H/A.
6. Review correlation pairs as potential redundancy/dependence, not as evidence that one feature is accurate.
