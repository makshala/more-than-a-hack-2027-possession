"""Contract and smoke tests for the possession skeleton.

Synthetic data only: nothing here reads or needs the competition Parquet.
Written with ``unittest`` so it runs with plain ``python -m unittest`` and
also under pytest.

All numeric values marked TEST-ONLY exist solely to exercise interfaces. They
are NOT proposals for the real configuration.
"""

from __future__ import annotations

import importlib
import py_compile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import config as project_config
from contracts import (
    ABSTAIN_CODE, AWAY_CODE, HOME_CODE,
    ContractError, DeadBallProbabilities, HADProbabilities,
    LabelMatrix, LiveProbabilities, SoftLabels, team_code_to_vote,
)
from feature_engineering import build_feature_frame, expected_input_columns
from fusion_export import export_had, fuse_had, to_export_frame
from label_model import (
    GenerativeLabelModel, SnorkelBackend, LabelModelConfigError,
    lf_summary, pairwise_agreement,
)
from labeling_functions import (
    GeometryLF, LFConfigError, LFResult, PossessionHistoryLF, VideoTagLF,
    apply_labeling_functions, build_labeling_functions,
)
from quality_gating import compute_quality_scores
from student_model import (
    StudentConfigError, build_student, iter_group_ablations, select_feature_columns,
)

ROOT = Path(__file__).resolve().parent.parent
MODULES = (
    "config", "contracts", "feature_engineering", "quality_gating",
    "quality_diagnostics", "labeling_functions", "label_model",
    "student_model", "fusion_export",
)


# ---------------------------------------------------------------------------
# Synthetic data and TEST-ONLY configuration
# ---------------------------------------------------------------------------
def make_features(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Random 111-column frame that satisfies Quality Gating's domain checks."""
    rng = np.random.default_rng(seed)
    cols = expected_input_columns()
    df = pd.DataFrame(rng.uniform(0.0, 1.0, size=(n, len(cols))), columns=list(cols))
    df["frame_id"] = np.arange(1000, 1000 + n)
    df["period"] = 1
    df["is_graded"] = 1
    df["is_pre_kickoff_d"] = 0
    df["timestamp_s"] = np.arange(n) / 25.0
    df["ball_status"] = rng.choice(["OBSERVED", "INTERPOLATED_SHORT", "INFERRED_LATENT"], n)
    df["ball_status_code"] = df["ball_status"].map(
        {"OBSERVED": 0, "INTERPOLATED_SHORT": 1, "INFERRED_LATENT": 2}
    )
    df["ball_in_camera_fov"] = rng.integers(0, 2, n)
    df["ball_near_boundary"] = rng.integers(0, 2, n)
    df["ball_gap_frames"] = rng.integers(0, 30, n).astype(float)
    df["home_n_visible"] = rng.integers(8, 12, n).astype(float)
    df["away_n_visible"] = rng.integers(8, 12, n).astype(float)
    df["last_controlled_team"] = rng.choice([-1, 1], n)
    df["frames_since_last_control"] = rng.integers(0, 200, n).astype(float)
    for col in ("st_set_piece", "st_throw_in", "st_corner_kick", "st_goal_kick", "st_free_kick",
                "st_kick_off", "st_goal", "st_post_goal_dead", "st_defending_tag", "st_in_box_tag"):
        df[col] = rng.integers(0, 2, n)
    df["st_restart_team"] = rng.choice([-1, 0, 1], n)
    df["st_team_prior"] = rng.choice([-1, 0, 1], n)
    df["st_n_active_tags"] = rng.integers(0, 4, n).astype(float)
    df["brd_valid"] = 1
    df["brd_is_replay"] = 0
    return df


def make_test_config() -> dict:
    """Deep copy of CONFIG with TEST-ONLY values filled in."""
    cfg = project_config.get_config()
    q = cfg["quality_gating"]
    q["ball"]["status_quality"] = {"OBSERVED": 1.0, "INTERPOLATED_SHORT": 0.7, "INFERRED_LATENT": 0.3}
    q["ball"]["fov_quality"] = {0: 0.5, 1: 1.0}
    q["ball"]["sigma_scale_m"] = 1.0
    q["ball"]["gap_scale_frames"] = 10.0
    q["homography"]["reprojection_error_scale"] = 1.0
    q["history"]["age_decay_frames"] = 50.0
    q["broadcast"]["replay_recovery_frames"] = 25.0

    lf = cfg["labeling_functions"]
    lf["causality_policy"] = "allow_non_causal"
    lf["quality_cutoffs"] = {e: 0.05 for e in lf["experts"]}

    st = cfg["student_model"]
    st["features"].update(
        groups="all_predictors",
        causality_policy="allow_non_causal",
        exclude_columns=("ball_status",),  # string column; see select_feature_columns
        missing_value_policy="error",
    )
    st["logistic"].update(C=1.0, l1_ratio=0.0, standardize=True)
    st["ablation"]["scheme"] = "leave_one_group_out"
    return cfg


class _HistoryLF(PossessionHistoryLF):
    def _vote(self, features, quality):
        votes = team_code_to_vote(features["last_controlled_team"], self._root["project"]["team_codes"])
        return LFResult(votes, pd.DataFrame({"raw": votes}, index=features.index))


class _TagLF(VideoTagLF):
    def _vote(self, features, quality):
        votes = team_code_to_vote(features["st_restart_team"], self._root["project"]["team_codes"])
        return LFResult(votes)


class _GeometryLF(GeometryLF):
    def _vote(self, features, quality):
        votes = np.where(features["home_pitch_control"] > 0.5, HOME_CODE, AWAY_CODE).astype("int8")
        return LFResult(votes)


def _fid(n):
    return np.arange(n)


# ---------------------------------------------------------------------------
class TestRepositoryHygiene(unittest.TestCase):
    def test_syntax_compiles(self):
        for name in MODULES:
            py_compile.compile(str(ROOT / f"{name}.py"), doraise=True)

    def test_all_modules_import(self):
        for name in MODULES:
            importlib.import_module(name)


class TestConfig(unittest.TestCase):
    def test_structure_valid(self):
        project_config.validate_project_config()
        project_config.validate_pipeline_structure()

    def test_structure_detects_missing_key(self):
        cfg = project_config.get_config()
        del cfg["fusion_export"]["output_format"]
        with self.assertRaises(ValueError):
            project_config.validate_pipeline_structure(cfg)

    def test_unresolved_register_lists_todos(self):
        todo = project_config.unresolved_parameters()
        self.assertIn("labeling_functions.causality_policy", todo)
        self.assertIn("student_model.logistic.C", todo)
        self.assertNotIn("student_model.logistic.class_weight", todo)

    def test_quality_params_unset_by_default(self):
        with self.assertRaises(ValueError):
            project_config.validate_quality_gating_config()

    def test_get_config_is_independent_copy(self):
        cfg = project_config.get_config()
        cfg["label_model"]["backend"] = "changed"
        self.assertEqual(project_config.CONFIG["label_model"]["backend"], "snorkel")


class TestLabelMatrix(unittest.TestCase):
    def test_valid(self):
        m = LabelMatrix(np.array([[0, -1], [1, 1]], dtype="int8"), ("a", "b"), _fid(2))
        self.assertEqual((m.n_frames, m.n_lfs), (2, 2))
        self.assertEqual(m.values.dtype, np.int8)

    def test_invalid_code_rejected(self):
        with self.assertRaises(ContractError):
            LabelMatrix(np.array([[2]]), ("a",), _fid(1))

    def test_float_dtype_rejected(self):
        with self.assertRaises(ContractError):
            LabelMatrix(np.array([[0.0]]), ("a",), _fid(1))

    def test_duplicate_names_and_ids_rejected(self):
        with self.assertRaises(ContractError):
            LabelMatrix(np.zeros((2, 2), dtype="int8"), ("a", "a"), _fid(2))
        with self.assertRaises(ContractError):
            LabelMatrix(np.zeros((2, 1), dtype="int8"), ("a",), np.array([1, 1]))

    def test_diagnostics_shapes(self):
        m = LabelMatrix(np.array([[0, 1], [0, -1], [-1, -1]], dtype="int8"), ("a", "b"), _fid(3))
        s = lf_summary(m)
        self.assertEqual(list(s.index), ["a", "b"])
        self.assertAlmostEqual(s.loc["a", "coverage"], 2 / 3)
        self.assertAlmostEqual(s.loc["a", "conflict"], 1 / 3)
        self.assertEqual(pairwise_agreement(m).shape, (2, 2))


class TestSoftLabels(unittest.TestCase):
    def _mat(self):
        return LabelMatrix(np.array([[0, 0], [0, 1], [1, -1], [-1, -1]], dtype="int8"), ("a", "b"), _fid(4))

    def test_format_from_majority_vote(self):
        cfg = make_test_config()
        cfg["label_model"]["backend"] = "majority_vote"
        soft = GenerativeLabelModel.from_config(cfg).fit_predict(self._mat())
        self.assertEqual(soft.probs.shape, (4, 2))
        np.testing.assert_allclose(soft.probs.sum(axis=1), 1.0)
        self.assertEqual(soft.covered.tolist(), [True, True, True, False])
        self.assertFalse(soft.metadata["calibrated"])
        np.testing.assert_allclose(soft.probs[1], [0.5, 0.5])

    def test_bad_probabilities_rejected(self):
        for probs in (np.array([[0.6, 0.6]]), np.array([[np.nan, 1.0]]), np.array([[1.2, -0.2]])):
            with self.assertRaises(ContractError):
                SoftLabels(_fid(1), probs, np.array([True]), "x")

    def test_weighted_hard_expansion(self):
        soft = SoftLabels(_fid(3), np.array([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]]),
                          np.array([True, False, True]), "x")
        rows, y, w = soft.expand_to_weighted_hard()
        self.assertEqual(sorted(set(rows.tolist())), [0, 2])
        np.testing.assert_allclose(w.sum(), 2.0)  # one unit of weight per covered row
        self.assertEqual(set(y.tolist()), {HOME_CODE, AWAY_CODE})

    def test_snorkel_backend_requires_params_before_anything(self):
        with self.assertRaises(LabelModelConfigError):
            SnorkelBackend({"n_epochs": None, "lr": None, "l2": None, "seed": None})


class TestLabelingFunctions(unittest.TestCase):
    def setUp(self):
        self.cfg = make_test_config()
        self.features = make_features(60)
        self.quality = compute_quality_scores(self.features, self.cfg["quality_gating"])

    def test_registry_builds_all_experts(self):
        lfs = build_labeling_functions(self.cfg)
        self.assertEqual([lf.name for lf in lfs], list(self.cfg["labeling_functions"]["experts"]))

    def test_skeleton_experts_not_implemented(self):
        for lf in build_labeling_functions(self.cfg):
            with self.assertRaises(NotImplementedError):
                lf.apply(self.features, self.quality)

    def test_refuses_without_cutoff_or_policy(self):
        cfg = make_test_config()
        cfg["labeling_functions"]["quality_cutoffs"] = {}
        with self.assertRaises(LFConfigError):
            _HistoryLF(cfg).apply(self.features, self.quality)
        cfg = make_test_config()
        cfg["labeling_functions"]["causality_policy"] = None
        with self.assertRaises(LFConfigError):
            _HistoryLF(cfg).apply(self.features, self.quality)

    def test_causal_only_blocks_non_causal_expert(self):
        cfg = make_test_config()
        cfg["labeling_functions"]["causality_policy"] = "causal_only"
        lf = _HistoryLF(cfg)
        self.assertTrue(lf.non_causal_columns)  # control_diff_roll50/100
        with self.assertRaises(LFConfigError):
            lf.apply(self.features, self.quality)

    def test_quality_gate_forces_abstain(self):
        quality = self.quality.copy()
        quality.loc[quality.index[:5], "q_history"] = 0.0
        res = _HistoryLF(self.cfg).apply(self.features, quality)
        self.assertTrue((res.votes[:5] == ABSTAIN_CODE).all())
        self.assertIn("possession_history__raw", res.debug.columns)

    def test_label_matrix_assembly_and_order(self):
        lfs = [_HistoryLF(self.cfg), _TagLF(self.cfg), _GeometryLF(self.cfg)]
        matrix, debug = apply_labeling_functions(self.features, self.quality, lfs)
        self.assertEqual(matrix.values.shape, (60, 3))
        np.testing.assert_array_equal(matrix.frame_id, self.features["frame_id"].to_numpy())
        self.assertIsNotNone(debug)


class TestStudent(unittest.TestCase):
    def setUp(self):
        self.cfg = make_test_config()
        self.features = make_features(300, seed=1)
        rng = np.random.default_rng(2)
        p_h = rng.uniform(0.05, 0.95, 300)
        self.soft = SoftLabels(self.features["frame_id"].to_numpy(), np.column_stack([p_h, 1 - p_h]),
                               rng.uniform(size=300) < 0.7, "test")

    def test_feature_selection_rules(self):
        cols = select_feature_columns(self.features, self.cfg)
        self.assertNotIn("frame_id", cols)
        self.assertNotIn("period", cols)
        self.assertNotIn("ball_status", cols)
        self.assertIn("near1_diff_leadlag_10f", cols)
        self.cfg["student_model"]["features"]["causality_policy"] = "causal_only"
        self.assertNotIn("near1_diff_leadlag_10f", select_feature_columns(self.features, self.cfg))

    def test_string_predictor_fails_loudly(self):
        self.cfg["student_model"]["features"]["exclude_columns"] = ()
        with self.assertRaises(StudentConfigError):
            select_feature_columns(self.features, self.cfg)

    def test_unset_config_fails_loudly(self):
        with self.assertRaises(StudentConfigError):
            select_feature_columns(self.features, project_config.get_config())
        cfg = make_test_config()
        cfg["student_model"]["logistic"]["C"] = None
        with self.assertRaises(StudentConfigError):
            build_student("logistic", cfg).fit(self.features, self.soft)

    def test_logistic_contract(self):
        model = build_student("logistic", self.cfg).fit(self.features, self.soft)
        live = model.predict_live(self.features)
        self.assertIsInstance(live, LiveProbabilities)
        self.assertEqual(live.probs.shape, (300, 2))
        np.testing.assert_array_equal(live.frame_id, self.features["frame_id"].to_numpy())
        self.assertEqual(len(model.explain()), len(model.feature_names_))

    def test_misaligned_frame_ids_rejected(self):
        shuffled = self.features.sample(frac=1.0, random_state=0)
        with self.assertRaises(ContractError):
            build_student("logistic", self.cfg).fit(shuffled, self.soft)

    def test_nan_predictors_rejected(self):
        bad = self.features.copy()
        bad.loc[0, "ball_x"] = np.nan
        with self.assertRaises(ContractError):
            build_student("logistic", self.cfg).fit(bad, self.soft)

    def test_gam_is_skeleton(self):
        with self.assertRaises(StudentConfigError):
            build_student("gam", self.cfg).fit(self.features, self.soft)
        self.cfg["student_model"]["gam"].update(library="x", n_splines=1, smoothing=1)
        with self.assertRaises(NotImplementedError):
            build_student("gam", self.cfg).fit(self.features, self.soft)

    def test_group_ablation_plan(self):
        plan = list(iter_group_ablations(self.cfg))
        self.assertEqual(len(plan), 11)
        self.assertTrue(all(len(g) == 10 for _, g in plan))


class TestFusionExport(unittest.TestCase):
    def setUp(self):
        self.cfg = make_test_config()
        self.ids = np.array([30, 10, 20, 40])  # deliberately not sorted
        self.live = LiveProbabilities(self.ids, np.array([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5], [1.0, 0.0]]), "m")

    def test_formula_and_sum(self):
        dead = DeadBallProbabilities(self.ids, np.array([0.0, 0.5, 1.0, 0.2]))
        had = fuse_had(self.live, dead, self.cfg)
        np.testing.assert_allclose(had.probs.sum(axis=1), 1.0)
        np.testing.assert_allclose(had.probs[1], [0.5 * 0.2, 0.5 * 0.8, 0.5])
        np.testing.assert_allclose(had.probs[2], [0.0, 0.0, 1.0])
        self.assertTrue(((had.probs >= 0) & (had.probs <= 1)).all())

    def test_dead_ball_reordered_by_frame_id_not_position(self):
        order = np.array([2, 0, 3, 1])
        p = np.array([0.1, 0.2, 0.3, 0.4])  # P(D) by id [30, 10, 20, 40]
        dead = DeadBallProbabilities(self.ids[order], p[order])
        had = fuse_had(self.live, dead, self.cfg)
        np.testing.assert_array_equal(had.frame_id, self.ids)
        np.testing.assert_allclose(had.probs[:, 2], p)

    def test_invalid_dead_ball_rejected(self):
        for bad in (np.array([np.nan, 0, 0, 0]), np.array([1.1, 0, 0, 0]), np.array([-0.1, 0, 0, 0])):
            with self.assertRaises(ContractError):
                DeadBallProbabilities(self.ids, bad)

    def test_frame_id_mismatch_rejected(self):
        dead = DeadBallProbabilities(np.array([30, 10, 20, 99]), np.zeros(4))
        with self.assertRaises(ContractError):
            fuse_had(self.live, dead, self.cfg)
        with self.assertRaises(ContractError):
            DeadBallProbabilities(np.array([1, 1, 2, 3]), np.zeros(4))

    def test_hadprobabilities_rejects_bad_sum(self):
        with self.assertRaises(ContractError):
            HADProbabilities(self.ids, np.full((4, 3), 0.4))

    def test_export_csv_preserves_order_and_columns(self):
        dead = DeadBallProbabilities(self.ids, np.array([0.1, 0.2, 0.3, 0.4]))
        had = fuse_had(self.live, dead, self.cfg)
        self.cfg["fusion_export"]["output_format"] = "csv"
        with tempfile.TemporaryDirectory() as tmp:
            path = export_had(had, Path(tmp) / "out.csv", self.cfg)
            table = pd.read_csv(path)
        self.assertEqual(list(table.columns), ["frame_id", "p_home", "p_away", "p_dead_ball"])
        np.testing.assert_array_equal(table["frame_id"].to_numpy(), self.ids)
        np.testing.assert_allclose(table[["p_home", "p_away", "p_dead_ball"]].sum(axis=1), 1.0)

    def test_export_parquet_if_engine_available(self):
        try:
            importlib.import_module("pyarrow")
        except ImportError:
            self.skipTest("pyarrow not installed in this environment")
        dead = DeadBallProbabilities(self.ids, np.array([0.1, 0.2, 0.3, 0.4]))
        had = fuse_had(self.live, dead, self.cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = export_had(had, Path(tmp) / "out.parquet", self.cfg)
            table = pd.read_parquet(path)
        np.testing.assert_array_equal(table["frame_id"].to_numpy(), self.ids)

    def test_to_export_frame_rejects_duplicate_column_names(self):
        had = HADProbabilities(self.ids, np.tile([0.2, 0.3, 0.5], (4, 1)))
        self.cfg["fusion_export"]["p_home"] = None
        self.cfg["fusion_export"]["away_output_column"] = "p_home"
        with self.assertRaises(ValueError):
            to_export_frame(had, self.cfg)


class TestEndToEndSynthetic(unittest.TestCase):
    """Whole chain on synthetic data, using TEST-ONLY LFs and the baseline backend."""

    def test_pipeline_runs_and_satisfies_had_contract(self):
        cfg = make_test_config()
        cfg["label_model"]["backend"] = "majority_vote"
        raw = make_features(500, seed=3)

        features = build_feature_frame(raw, allow_extra_columns=False)       # stage 1
        quality = compute_quality_scores(features, cfg["quality_gating"])     # stage 2
        lfs = [_HistoryLF(cfg), _TagLF(cfg), _GeometryLF(cfg)]
        matrix, _ = apply_labeling_functions(features, quality, lfs)          # stage 3
        soft = GenerativeLabelModel.from_config(cfg).fit_predict(matrix)      # stage 4
        student = build_student("logistic", cfg).fit(features, soft)          # stage 5
        live = student.predict_live(features)

        rng = np.random.default_rng(4)
        dead = DeadBallProbabilities(features["frame_id"].to_numpy()[::-1].copy(), rng.uniform(0, 1, 500))
        had = fuse_had(live, dead, cfg)                                       # stage 6

        np.testing.assert_array_equal(had.frame_id, features["frame_id"].to_numpy())
        self.assertTrue(np.isfinite(had.probs).all())
        self.assertTrue(((had.probs >= 0) & (had.probs <= 1)).all())
        np.testing.assert_allclose(had.probs.sum(axis=1), 1.0, atol=1e-9)
        self.assertFalse(set(raw.columns) - set(features.columns))  # source columns preserved


if __name__ == "__main__":
    unittest.main()
