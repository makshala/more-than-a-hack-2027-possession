import numpy as np
import pandas as pd

# Importujemy wasze moduły
from main import _smooth_1d_reflect, _clean_segments_spell_aware

def run_oof_latency_optimization(csv_path="role4_artifacts/role4_oof_predictions.csv"):
    print(f"Wczytywanie predykcji OOF SkillCorner z: {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Automatyczne wykrywanie kolumn (odporność na długie przedrostki)
    p_home_col = next(c for c in df.columns if 'p_home' in c.lower())
    p_away_col = next(c for c in df.columns if 'p_away' in c.lower())
    
    # Próba znalezienia kolumny dead, a jeśli jest ucięta - bierzemy ostatnią pasującą do prawdopodobieństw
    try:
        p_dead_col = next(c for c in df.columns if 'dead' in c.lower())
    except StopIteration:
        prob_cols = [c for c in df.columns if 'p_' in c.lower() or 'regres' in c.lower()]
        p_dead_col = [c for c in prob_cols if c not in (p_home_col, p_away_col)][0]

    print(f"Wykryto kolumny:\n  H: {p_home_col}\n  A: {p_away_col}\n  D: {p_dead_col}")

    # =========================================================================
    # SKALOWANIE FPS (10 FPS SkillCorner -> 25 FPS Mecz docelowy)
    # Aby dekoder zachował się identycznie, musimy proporcjonalnie 
    # zmniejszyć okna czasowe i progi z main.py (współczynnik 10/25 = 0.4)
    # =========================================================================
    scale = 10.0 / 25.0
    win_ha_10fps = max(3, int(33 * scale)) # 33 -> 13
    win_d_10fps = max(3, int(11 * scale))  # 11 -> 4 (zaokr. 5 dla nieparzystości)
    win_d_10fps = 5 
    
    min_live_10fps = int(46 * scale) # 46 -> 18
    min_dead_10fps = int(40 * scale) # 40 -> 16
    
    shift_candidates = list(range(-5, 10)) # Przesunięcia do testowania w skali 10 FPS
    match_ids = df['match_id'].unique()
    shift_scores = {s: [] for s in shift_candidates}

    print(f"\nUruchamianie zoptymalizowanego dekodera na {len(match_ids)} meczach OOF...")

    for mid in match_ids:
        df_match = df[df['match_id'] == mid].sort_values('frame').copy()
        
        p_h = df_match[p_home_col].to_numpy(dtype=np.float64)
        p_a = df_match[p_away_col].to_numpy(dtype=np.float64)
        p_d = df_match[p_dead_col].to_numpy(dtype=np.float64)
        gt_states = df_match['target'].to_numpy(dtype=np.int32)

        # Odzyskiwanie D (przeskalowane konwolucje)
        d_core = p_d >= 0.48
        d_near_core = np.convolve(d_core.astype(np.int32), np.ones(15, dtype=np.int32), mode="same") > 0
        d_boosted = np.where(d_near_core & (p_d >= 0.28), np.maximum(p_d * 1.45, 0.54), p_d)
        is_d_modal = (p_d >= p_h) & (p_d >= p_a) & (p_d >= 0.36)
        d_boosted = np.where(is_d_modal, np.maximum(d_boosted, 0.53), d_boosted)

        # Wygładzanie (na parametrach dla 10 FPS)
        p_h_smooth = _smooth_1d_reflect(p_h, window=win_ha_10fps)
        p_a_smooth = _smooth_1d_reflect(p_a, window=win_ha_10fps)
        p_d_smooth = _smooth_1d_reflect(d_boosted, window=win_d_10fps)

        probs = np.stack([p_h_smooth, p_a_smooth, p_d_smooth], axis=1)
        probs = np.clip(probs, 0.02, 0.98)
        probs /= np.sum(probs, axis=1, keepdims=True)
        log_emit = np.log(probs)

        # Viterbi (Kary przechodzą bez zmian)
        n = len(df_match)
        pen_ha = 11.345
        pen_hd = 8.288
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

        # Cleanup (przeskalowany na 10 FPS)
        raw_pred_states = _clean_segments_spell_aware(
            states=states, probs=probs,
            min_live_frames=min_live_10fps,
            min_dead_frames=min_dead_10fps,
            live_sandwich_frames=min_live_10fps,
        )

        # Testowanie przesunięć
        for shift in shift_candidates:
            if shift == 0:
                shifted = raw_pred_states.copy()
            else:
                shifted = np.empty_like(raw_pred_states)
                if shift > 0:
                    shifted[:-shift] = raw_pred_states[shift:]
                    shifted[-shift:] = raw_pred_states[-1]
                else:
                    abs_shift = abs(shift)
                    shifted[abs_shift:] = raw_pred_states[:-abs_shift]
                    shifted[:abs_shift] = raw_pred_states[0]
            
            # Liczymy F1 względem targetu
            macro_f1 = np.mean(shifted == gt_states)
            shift_scores[shift].append(macro_f1)

    # Agregacja wyników
    avg_scores = {s: np.mean(scores) for s, scores in shift_scores.items()}
    best_shift_10fps = max(avg_scores, key=avg_scores.get)
    
    # Przeliczenie na 25 FPS (Mecz docelowy)
    best_shift_25fps = int(round(best_shift_10fps * (25.0 / 10.0)))

    print("\n" + "="*60)
    print(f"[WYNIK] Optymalne przesunięcie (10 FPS OOF): {best_shift_10fps} klatek")
    print(f"[WYNIK] Ekwiwalent przesunięcia (25 FPS DOCELOWE): {best_shift_25fps} klatek")
    print("="*60)

if __name__ == "__main__":
    # Upewnij się, że ścieżka do pliku OOF jest poprawna
    run_oof_latency_optimization(csv_path="role4_artifacts/role4_oof_predictions.csv")