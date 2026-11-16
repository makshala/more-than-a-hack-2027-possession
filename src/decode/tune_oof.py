import numpy as np
import pandas as pd
import optuna
import warnings

# Ignorowanie ostrzeżeń ze względu na czystość wyjścia
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from main import _smooth_1d_reflect, _clean_segments_spell_aware

def count_spells(states_array):
    if len(states_array) == 0: return 0
    return int(np.sum(states_array[1:] != states_array[:-1]) + 1)

def objective(trial, df, p_home_col, p_away_col, p_dead_col):
    # Szukamy hiperparametrów dla skali SkillCornera (10 FPS)
    shift = trial.suggest_int('shift', 0, 5) # Opóźnienie 0 do 0.5s
    win_ha = trial.suggest_int('win_ha', 5, 19, step=2) 
    win_d = trial.suggest_int('win_d', 3, 9, step=2)
    pen_ha = trial.suggest_float('pen_ha', 5.0, 15.0)
    pen_hd = trial.suggest_float('pen_hd', 4.0, 12.0)
    min_live = trial.suggest_int('min_live', 12, 25)
    
    match_ids = df['match_id'].unique()
    scores = []
    
    # Trenujemy na 5 pierwszych meczach OOF dla szybkości eksperymentu
    for mid in match_ids[:5]:
        df_match = df[df['match_id'] == mid].sort_values('frame').copy()
        
        p_h = df_match[p_home_col].to_numpy(dtype=np.float64)
        p_a = df_match[p_away_col].to_numpy(dtype=np.float64)
        p_d = df_match[p_dead_col].to_numpy(dtype=np.float64)
        gt_states = df_match['target'].to_numpy(dtype=np.int32)
        
        # --- NAUKOWY TIME-SHIFTING NA PRAWDOPODOBIEŃSTWACH ---
        # Cofamy wejściowe odczyty w czasie, aby zrekompensować opóźnienie modelu
        if shift > 0:
            p_h = np.pad(p_h[shift:], (0, shift), mode='edge')
            p_a = np.pad(p_a[shift:], (0, shift), mode='edge')
            p_d = np.pad(p_d[shift:], (0, shift), mode='edge')

        # Rekonstrukcja D (dla 10 FPS)
        d_core = p_d >= 0.48
        d_near_core = np.convolve(d_core.astype(np.int32), np.ones(15, dtype=np.int32), mode="same") > 0
        d_boosted = np.where(d_near_core & (p_d >= 0.28), np.maximum(p_d * 1.45, 0.54), p_d)
        is_d_modal = (p_d >= p_h) & (p_d >= p_a) & (p_d >= 0.36)
        d_boosted = np.where(is_d_modal, np.maximum(d_boosted, 0.53), d_boosted)

        # Wygładzanie
        p_h_smooth = _smooth_1d_reflect(p_h, window=win_ha)
        p_a_smooth = _smooth_1d_reflect(p_a, window=win_ha)
        p_d_smooth = _smooth_1d_reflect(d_boosted, window=win_d)

        probs = np.stack([p_h_smooth, p_a_smooth, p_d_smooth], axis=1)
        probs = np.clip(probs, 0.02, 0.98)
        probs /= np.sum(probs, axis=1, keepdims=True)
        log_emit = np.log(probs)

        # Viterbi
        n = len(df_match)
        log_trans = np.array([
            [0.0, -pen_ha, -pen_hd],
            [-pen_ha, 0.0, -pen_hd],
            [-pen_hd, -pen_hd, 0.0],
        ], dtype=np.float64)

        dp = np.zeros((n, 3), dtype=np.float64)
        back = np.zeros((n, 3), dtype=np.int32)
        dp[0] = log_emit[0]

        for t in range(1, n):
            scores_dp = dp[t - 1][:, None] + log_trans
            back[t] = np.argmax(scores_dp, axis=0)
            dp[t] = np.max(scores_dp, axis=0) + log_emit[t]

        states = np.zeros(n, dtype=np.int32)
        states[-1] = int(np.argmax(dp[-1]))
        for t in range(n - 2, -1, -1):
            states[t] = back[t + 1, states[t + 1]]

        # Cleanup
        states_clean = _clean_segments_spell_aware(
            states=states, probs=probs,
            min_live_frames=min_live,
            min_dead_frames=min_live-2,
            live_sandwich_frames=min_live,
        )
        
        macro_f1 = np.mean(states_clean == gt_states)
        gt_spells = count_spells(gt_states)
        pred_spells = count_spells(states_clean)
        frag = pred_spells / max(1, gt_spells)
        
        # Bardzo rygorystyczna kara za fragmentację (surowsza niż wcześniej)
        frag_penalty = 1.5 * abs(frag - 1.0)
        scores.append(macro_f1 - frag_penalty)
        
    return np.mean(scores)

if __name__ == "__main__":
    print("Inicjalizacja naukowej optymalizacji z wbudowanym Phase Shiftem (OOF)...")
    df = pd.read_csv("role4_artifacts/role4_oof_predictions.csv")
    
    # Detekcja kolumn
    p_home_col = next(c for c in df.columns if 'p_home' in c.lower())
    p_away_col = next(c for c in df.columns if 'p_away' in c.lower())
    prob_cols = [c for c in df.columns if 'p_' in c.lower() or 'regres' in c.lower()]
    p_dead_col = [c for c in prob_cols if c not in (p_home_col, p_away_col)][0]

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective(t, df, p_home_col, p_away_col, p_dead_col), n_trials=75, show_progress_bar=True)
    
    print("\n" + "="*50)
    print("OPTYMALNE PARAMETRY DLA OOF (SKALA 10 FPS):")
    best = study.best_params
    for k, v in best.items(): 
        print(f"  {k}: {v}")
    print("="*50)
    print("Proszę, wklej mi powyższe wyniki. Wykonam wtedy bezpieczną przesiadkę do main.py.")