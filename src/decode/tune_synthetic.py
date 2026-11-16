import os
import sys
import numpy as np
import pandas as pd
import optuna

# Dodanie ścieżki, aby poprawnie zaimportować moduły Roli 5
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from generator import SyntheticPossessionGenerator
from main import _smooth_1d_reflect, _clean_segments_spell_aware

# Stałe stanów
STATE_HOME = 0
STATE_AWAY = 1
STATE_DEAD_BALL = 2

def count_spells(states_array: np.ndarray) -> int:
    """Zwraca liczbę ciągłych segmentów (spelli) w wektorze stanów."""
    if len(states_array) == 0:
        return 0
    return int(np.sum(states_array[1:] != states_array[:-1]) + 1)

def parameterized_decode(df_half: pd.DataFrame, params: dict) -> np.ndarray:
    """Kopia logiki z main.py, ale z wstrzykiwanymi hiperparametrami z Optuny."""
    p_h = df_half["p_home"].to_numpy(dtype=np.float64)
    p_a = df_half["p_away"].to_numpy(dtype=np.float64)
    dead_col = "p_dead" if "p_dead" in df_half.columns else "p_dead_ball"
    p_d = df_half[dead_col].to_numpy(dtype=np.float64)

    # 1. Odzyskiwanie Dead-Ball
    d_core = p_d >= 0.48
    d_near_core = np.convolve(d_core.astype(np.int32), np.ones(37, dtype=np.int32), mode="same") > 0
    d_boosted = np.where(d_near_core & (p_d >= 0.28), np.maximum(p_d * 1.45, 0.54), p_d)
    is_d_modal = (p_d >= p_h) & (p_d >= p_a) & (p_d >= 0.36)
    d_boosted = np.where(is_d_modal, np.maximum(d_boosted, 0.53), d_boosted)

    # 2. Wygładzanie (parametryzowane)
    p_h_smooth = _smooth_1d_reflect(p_h, window=params['win_ha'])
    p_a_smooth = _smooth_1d_reflect(p_a, window=params['win_ha'])
    p_d_smooth = _smooth_1d_reflect(d_boosted, window=params['win_d'])

    probs = np.stack([p_h_smooth, p_a_smooth, p_d_smooth], axis=1)
    probs = np.clip(probs, 0.02, 0.98)
    probs /= np.sum(probs, axis=1, keepdims=True)
    log_emit = np.log(probs)

    # 3. Dekoder Viterbiego (parametryzowane kary)
    n = len(df_half)
    pen_ha = params['pen_ha']
    pen_hd = params['pen_hd']
    
    log_trans = np.array([
        [0.0, -pen_ha, -pen_hd],
        [-pen_ha, 0.0, -pen_hd],
        [-pen_hd, -pen_hd, 0.0],
    ], dtype=np.float64)

    dp = np.zeros((n, 3), dtype=np.float64)
    back = np.zeros((n, 3), dtype=np.int32)
    dp[0] = log_emit[0]

    for t in range(1, n):
        scores = dp[t - 1][:, None] + log_trans
        back[t] = np.argmax(scores, axis=0)
        dp[t] = np.max(scores, axis=0) + log_emit[t]

    states = np.zeros(n, dtype=np.int32)
    states[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]

    # 4. Spell-aware cleanup (parametryzowane progi)
    states_clean = _clean_segments_spell_aware(
        states=states,
        probs=probs,
        min_live_frames=params['min_live_frames'],
        min_dead_frames=params['min_dead_frames'],
        live_sandwich_frames=params['live_sandwich_frames'],
    )
    return states_clean

def objective(trial):
    # Przestrzeń poszukiwań
    params = {
        'win_ha': trial.suggest_int('win_ha', 21, 61, step=2),       # Od 0.8s do 2.4s
        'win_d': trial.suggest_int('win_d', 9, 21, step=2),          # Zostawiamy w miarę ostre dla D
        'pen_ha': trial.suggest_float('pen_ha', 8.0, 15.0),          # Karyfikacja nerwowych zmian
        'pen_hd': trial.suggest_float('pen_hd', 5.0, 10.0),
        'min_live_frames': trial.suggest_int('min_live_frames', 25, 60), # 1.0s do 2.4s fizycznej kontroli
        'min_dead_frames': trial.suggest_int('min_dead_frames', 20, 40),
        'live_sandwich_frames': trial.suggest_int('live_sandwich_frames', 20, 50),
    }

    scores = []
    # Ewaluacja na 3 różnych, niezależnie wygenerowanych scenariuszach syntetycznych
    for seed in [42, 123, 999]:
        df_sim = SyntheticPossessionGenerator.generate_tactical_match_simulation(
            num_spells=150, 
            fps=25.0, 
            base_confidence=0.85, 
            random_seed=seed
        )
        
        gt_states = df_sim["ground_truth"].to_numpy(dtype=np.int32)
        gt_spells_count = count_spells(gt_states)
        
        # Predykcja
        pred_states = parameterized_decode(df_sim, params)
        pred_spells_count = count_spells(pred_states)
        
        # Metryki
        macro_f1 = np.mean(pred_states == gt_states)
        frag = pred_spells_count / max(1, gt_spells_count)
        
        # Funkcja celu: Maksymalizuj MacroF1, bardzo surowo karz odchylenia frag od 1.0
        frag_penalty = 2.0 * abs(frag - 1.0)
        score = macro_f1 - frag_penalty
        scores.append(score)

    return np.mean(scores)

if __name__ == "__main__":
    print("Rozpoczynam optymalizację hiperparametrów Roli 5 na danych syntetycznych...")
    # Wyłączenie logów informacyjnych Optuny dla czystości konsoli
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    print("\n[SUKCES] Znaleziono optymalne parametry minimalizujące fragmentację:")
    best = study.best_params
    for key, value in best.items():
        print(f"  {key}: {value}")