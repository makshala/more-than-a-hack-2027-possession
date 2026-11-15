"""
Unit and integration test suite validating the possession temporal decoding ecosystem.
"""
import unittest
import numpy as np
import pandas as pd
from config import (
    ms_to_frames, frames_to_ms, safe_log, safe_log_sum_exp, LOG_EPS,
    STATE_HOME, STATE_AWAY, STATE_DEAD_BALL
)
from generator import SyntheticPossessionGenerator, REGIME_OCCLUSION, REGIME_CLUSTERING
from viterbi_decoder import ViterbiTemporalDecoder
from hsmm_decoder import HSMMExplicitDurationDecoder, ContinuousDurationDistribution
from spells import PossessionSpellExtractor
from diagnostics import TemporalDiagnostics


class TestPossessionDecodingEcosystem(unittest.TestCase):

    def setUp(self):
        self.fps = 25.0
        self.test_segments = [(0, 30), (2, 25), (1, 35), (0, 30)]
        self.df_data = SyntheticPossessionGenerator.generate_controlled_scenario(
            segments=self.test_segments,
            fps=self.fps,
            base_confidence=0.88,
            random_seed=42
        )
        self.A = np.array([
            [0.96, 0.01, 0.03],
            [0.01, 0.96, 0.03],
            [0.05, 0.05, 0.90]
        ], dtype=np.float64)

    def test_config_temporal_conversion(self):
        # 1000 ms at 25 fps must be 25 frames
        self.assertEqual(ms_to_frames(1000.0, fps=25.0), 25)
        self.assertEqual(frames_to_ms(25, fps=25.0), 1000.0)
        # Variable fps: 1000 ms at 50 fps must be 50 frames
        self.assertEqual(ms_to_frames(1000.0, fps=50.0), 50)

    def test_safe_log_boundaries(self):
        # Underflow safety
        self.assertGreater(safe_log(1e-25), -1e12)
        self.assertEqual(safe_log(0.0), -1e9)
        # Log-Sum-Exp safety
        arr = np.array([-1e9, 0.0, -1e9])
        lse = safe_log_sum_exp(arr)
        self.assertAlmostEqual(lse, 0.0, places=4)

    def test_generator_markovian_regimes(self):
        df_sim = SyntheticPossessionGenerator.generate_tactical_match_simulation(
            num_spells=5, fps=25.0, random_seed=42
        )
        self.assertIn("noise_regime", df_sim.columns)
        self.assertIn("ground_truth", df_sim.columns)
        self.assertTrue(len(df_sim) > 0)
        # Sum of probabilities must strictly equal 1.0
        prob_sums = df_sim[["p_home", "p_away", "p_dead_ball"]].sum(axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)

    def test_viterbi_bayesian_inversion(self):
        decoder = ViterbiTemporalDecoder(
            transition_matrix=self.A,
            input_semantics="posterior",
            entropy_factor=0.5
        )
        decoded = decoder.decode(self.df_data)
        self.assertEqual(len(decoded), len(self.df_data))
        self.assertTrue(set(np.unique(decoded)).issubset({0, 1, 2}))

    def test_hsmm_continuous_duration_trellis(self):
        hsmm = HSMMExplicitDurationDecoder(
            transition_matrix=self.A,
            duration_weight=1.0,
            input_semantics="posterior",
            fps=self.fps
        )
        decoded = hsmm.decode(self.df_data)
        self.assertEqual(len(decoded), len(self.df_data))
        # Ensure minimum duration is respected
        spells = PossessionSpellExtractor.extract_spells(self.df_data, decoded)
        self.assertTrue(len(spells) > 0)
        # Verify min duration in frames >= 5
        self.assertGreaterEqual(spells["duration_frames"].min(), 5)

    def test_morphological_filtering_and_hysteresis(self):
        raw_states = np.array([0, 0, 0, 1, 0, 0, 0, 2, 2, 2, 2], dtype=np.int32)
        # Single frame spike at index 3 should be eliminated by opening/closing
        filtered = PossessionSpellExtractor.apply_morphological_filtering(
            raw_states, closing_ms=300.0, opening_ms=100.0, fps=25.0
        )
        self.assertEqual(filtered[3], 0)

    def test_temporal_iou_and_spell_f1(self):
        gt_states = self.df_data["ground_truth"].to_numpy()
        df_gt_spells = PossessionSpellExtractor.extract_spells(self.df_data, gt_states)
        # Perfect matching gives SpellF1 = 1.0 and fragmentation penalty = 1.0
        metrics = TemporalDiagnostics.evaluate_spells(df_gt_spells, df_gt_spells)
        self.assertAlmostEqual(metrics["spell_f1"], 1.0, places=4)
        self.assertAlmostEqual(metrics["spell_f1_penalized"], 1.0, places=4)
        self.assertEqual(metrics["excess_fragments"], 0)


    def test_sandwich_micro_possession_merging(self):
        # Scenario: H H H H H A A A H H H H H (sandwich of 3 frames of A inside H)
        # 3 frames of A is 120ms at 25 fps, which should be merged into H
        raw_states = np.array([0]*30 + [1]*3 + [0]*30, dtype=np.int32)
        cleaned = PossessionSpellExtractor.suppress_micro_possessions(
            raw_states, fps=25.0, sandwich_max_ms=1600.0
        )
        # All 63 frames must now be 0 (Home)
        self.assertTrue(np.all(cleaned == 0))

        # Scenario with Dead Ball: A A A D D A A A (short 2-frame dead ball inside Away)
        raw_states2 = np.array([1]*25 + [2]*2 + [1]*25, dtype=np.int32)
        cleaned2 = PossessionSpellExtractor.suppress_micro_possessions(
            raw_states2, fps=25.0, sandwich_max_ms=1600.0
        )
        self.assertTrue(np.all(cleaned2 == 1))

    def test_role4_parquet_integration(self):
        import os, json
        parquet_path = "role5_probabilities.parquet"
        if not os.path.exists(parquet_path):
            self.skipTest("role5_probabilities.parquet not found")
        
        df = pd.read_parquet(parquet_path)
        self.assertIn("p_dead", df.columns)
        self.assertIn("p_home", df.columns)
        self.assertIn("p_away", df.columns)
        self.assertGreater(len(df), 1000)

        # Test normalization
        from config import normalize_probabilities_df
        df_norm = normalize_probabilities_df(df.iloc[:500])
        self.assertIn("p_dead_ball", df_norm.columns)
        self.assertIn("timestamp", df_norm.columns)

        # Test Viterbi decoding on Role 4 data
        viterbi = ViterbiTemporalDecoder(transition_matrix=self.A, input_semantics="posterior")
        states = viterbi.decode(df_norm)
        self.assertEqual(len(states), 500)

        # Test spell extraction with phase
        sync_meta = {"half_1": {"start_frame": 100, "end_frame": 400}}
        spells = PossessionSpellExtractor.extract_spells(df_norm, states, sync_metadata=sync_meta)
        self.assertIn("half", spells.columns)
        self.assertIn("mean_confidence", spells.columns)


if __name__ == "__main__":
    unittest.main()
