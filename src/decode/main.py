"""
Main execution pipeline comparing transition matrices, HMM vs HSMM, and evaluating spells.
Executes and demonstrates tasks 1-8.
"""
import numpy as np
from generator import SyntheticPossessionGenerator
from viterbi_decoder import ViterbiTemporalDecoder
from hsmm_decoder import HSMMExplicitDurationDecoder
from spells import PossessionSpellExtractor
from diagnostics import TemporalDiagnostics


def run_pipeline():
    print("======================================================================")
    print(" ROLE 5: POSSESSION ESTIMATION & TEMPORAL DECODING BENCHMARK")
    print("======================================================================\n")

    # 1. Generate synthetic scenario: H -> D -> A -> H with noisy glitches
    # Scenario: 50 frames H, 40 frames D, 60 frames A, 50 frames H
    test_segments = [(0, 50), (2, 40), (1, 60), (0, 50)]
    df_data = SyntheticPossessionGenerator.generate_controlled_scenario(
        segments=test_segments, fps=25.0, base_confidence=0.85, noise_level=0.10, random_seed=42
    )
    print(f"[Generator] Generated {len(df_data)} synthetic frames with controlled noise.")

    # 2. Compare transition matrices (Task 4)
    # A_sticky: Strong inertia (favors staying in current state)
    A_sticky = np.array([
        [0.98, 0.01, 0.01],
        [0.01, 0.98, 0.01],
        [0.02, 0.02, 0.96]
    ])

    # A_flexible: High willingness to switch
    A_flexible = np.array([
        [0.70, 0.15, 0.15],
        [0.15, 0.70, 0.15],
        [0.20, 0.20, 0.60]
    ])

    # HMM Sticky Decoder
    hmm_sticky = ViterbiTemporalDecoder(transition_matrix=A_sticky, input_semantics="posterior")
    states_sticky = hmm_sticky.decode(df_data)
    spells_sticky = PossessionSpellExtractor.extract_spells(df_data, states_sticky)
    diag_sticky = TemporalDiagnostics.compute_diagnostics(df_data, states_sticky, spells_sticky)

    # HMM Flexible Decoder
    hmm_flex = ViterbiTemporalDecoder(transition_matrix=A_flexible, input_semantics="posterior")
    states_flex = hmm_flex.decode(df_data)
    spells_flex = PossessionSpellExtractor.extract_spells(df_data, states_flex)
    diag_flex = TemporalDiagnostics.compute_diagnostics(df_data, states_flex, spells_flex)

    # 3. HSMM Decoder with explicit minimum duration (Tasks 5 & 6)
    # Require at least 15 frames (~0.6s) per spell
    min_dur = {0: 15, 1: 15, 2: 10}
    max_dur = {0: 100, 1: 100, 2: 100}
    hsmm = HSMMExplicitDurationDecoder(
        transition_matrix=A_sticky,
        min_durations=min_dur,
        max_durations=max_dur,
        input_semantics="posterior"
    )
    states_hsmm = hsmm.decode(df_data)
    spells_hsmm = PossessionSpellExtractor.extract_spells(df_data, states_hsmm)
    diag_hsmm = TemporalDiagnostics.compute_diagnostics(df_data, states_hsmm, spells_hsmm)

    # Comparison summary table
    print("\n--- RESULTS COMPARISON (Task 4, 5, 6, 7) ---")
    print(f"{'Metric':<32} | {'HMM (Sticky)':<14} | {'HMM (Flexible)':<14} | {'HSMM (Min Dur)':<14}")
    print("-" * 82)
    print(f"{'Detected Spells Count':<32} | {diag_sticky['total_spells']:<14} | {diag_flex['total_spells']:<14} | {diag_hsmm['total_spells']:<14}")
    print(f"{'Accuracy vs Ground Truth':<32} | {diag_sticky['accuracy_vs_ground_truth']*100:.2f}%{'':<8} | {diag_flex['accuracy_vs_ground_truth']*100:.2f}%{'':<8} | {diag_hsmm['accuracy_vs_ground_truth']*100:.2f}%{'':<8}")
    print(f"{'Divergence from Argmax':<32} | {diag_sticky['divergence_from_argmax']*100:.2f}%{'':<8} | {diag_flex['divergence_from_argmax']*100:.2f}%{'':<8} | {diag_hsmm['divergence_from_argmax']*100:.2f}%{'':<8}")
    print(f"{'Mean Spell Duration (s)':<32} | {diag_sticky['mean_spell_duration_seconds']:<14.2f} | {diag_flex['mean_spell_duration_seconds']:<14.2f} | {diag_hsmm['mean_spell_duration_seconds']:<14.2f}")
    print(f"{'Min Spell Duration (s)':<32} | {diag_sticky['min_spell_duration_seconds']:<14.2f} | {diag_flex['min_spell_duration_seconds']:<14.2f} | {diag_hsmm['min_spell_duration_seconds']:<14.2f}")

    print("\n--- SAMPLE EXTRACTED SPELLS (HSMM) ---")
    print(spells_hsmm.to_string(index=False))


if __name__ == '__main__':
    run_pipeline()
