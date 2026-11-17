import importlib.util
import warnings
import numpy as np
import pandas as pd
import optuna
import os
import sys

# Ignorowanie ostrzeżeń ze względu na czystość wyjścia
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.abspath(os.path.join(HERE, "..", "possession", "inference"))

for path in (HERE, INFERENCE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from main import _smooth_1d_reflect, _clean_segments_spell_aware
from role4_evidence_fusion import apply_directional_evidence

def count_spells(states_array):
    if len(states_array) == 0:
        return 0
    return int(np.sum(states_array[1:] != states_array[:-1]) + 1)

def objective(trial, df, p_home_col, p_away_col, p_dead_col, train_mids, val_mids):
    # Szukamy hiperparametrów dla skali SkillCornera (10 FPS)
    shift = trial.suggest_int('shift', 0, 5)  # Opóźnienie 0 do 0.5s
    win_ha = trial.suggest_int('win_ha', 5, 11, step=2)
    win_d = trial.suggest_int('win_d', 4, 7, step=2)  # Poprawiona nazwa parametru!
    pen_ha = trial.suggest_float('pen_ha', 3.0, 8.0)
    pen_hd = trial.suggest_float('pen_hd', 3.0, 8.0)
    min_live = trial.suggest_int('min_live', 7, 16)
    
    train_scores = []
    val_scores = []
    
    all_current_mids = list(train_mids) + list(val_mids)
    
    for mid in all_current_mids:
        df_match = df[df['match_id'].astype(str) == str(mid)].sort_values('frame').copy()
        
        if len(df_match) == 0:
            continue
        
        p_h = df_match[p_home_col].to_numpy(dtype=np.float64)
        p_a = df_match[p_away_col].to_numpy(dtype=np.float64)
        p_d = df_match[p_dead_col].to_numpy(dtype=np.float64)
        gt_states = df_match['target'].to_numpy(dtype=np.int32)
        
        # --- NAUKOWY TIME-SHIFTING NA PRAWDOPODOBIEŃSTWACH ---
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
            min_dead_frames=max(1, min_live - 2),
            live_sandwich_frames=min_live,
        )
        
        macro_f1 = np.mean(states_clean == gt_states)
        gt_spells = count_spells(gt_states)
        pred_spells = count_spells(states_clean)
        frag = pred_spells / max(1, gt_spells)
        
        # POLUZOWANA kara za fragmentację 
        frag_penalty = 0.2 * abs(frag - 1.0)
        final_score = macro_f1 - frag_penalty
        
        if str(mid) in train_mids:
            train_scores.append(final_score)
        else:
            val_scores.append(final_score)
            
    mean_train = np.mean(train_scores) if train_scores else 0
    mean_val = np.mean(val_scores) if val_scores else 0
    
    trial.set_user_attr("train_score", float(mean_train))
    trial.set_user_attr("val_score", float(mean_val))
    
    return mean_val

if __name__ == "__main__":
    print("Inicjalizacja naukowej optymalizacji z chronionym testem i Evidence Fusion (OOF)...")
    
    # --- 1. WCZYTANIE KONFIGURACJI Z NOWYMI REGUŁAMI ---
    config_path = os.path.abspath(os.path.join(HERE, "..", "possession", "config.py"))
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    evidence_cfg = cfg_mod.get_config()["role4_evidence_fusion"]

    # --- 2. WCZYTANIE SUROWEGO OOF I CECH FIZYCZNYCH ---
    df_oof = pd.read_csv("role4_artifacts/role4_oof_predictions.csv")
    
    # PODAJ TUTAJ WŁAŚCIWĄ ŚCIEŻKĘ DO PARQUETA Z CECHAMI 20 MECZÓW:
    df_features = pd.read_parquet("src/possession/training/skillcorner_training.parquet") 

    df_oof["match_id"] = df_oof["match_id"].astype(str)
    df_features["match_id"] = df_features["match_id"].astype(str)

    # Dołączamy tylko cechy wymagane do nowych reguł Evidence Fusion
    # Bierzemy z df_features wszystkie kolumny, których nie ma jeszcze w df_oof (plus klucze do złączenia)
    extra_cols = ["match_id", "frame"] + [c for c in df_features.columns if c not in df_oof.columns]

    df = pd.merge(
        df_oof,
        df_features[extra_cols],
        on=["match_id", "frame"],
        how="inner",
        validate="one_to_one"
    )

    # Zabezpieczenie na wypadek, gdyby Parquet ze SkillCorner nie miał kolumny 'is_graded'
    if "is_graded" not in df.columns:
        df["is_graded"] = 1

    # --- 3. FILTROWANIE REGUŁ I APLIKACJA EVIDENCE FUSION ---
    valid_rules = []
    for rule in evidence_cfg.get("rules", []):
        if not rule.get("active", True):
            continue
        req_cols = [rule.get("column"), rule.get("home_column"), rule.get("away_column")]
        for gate in rule.get("quality_gates", []):
            req_cols.extend([gate.get("column"), gate.get("home_column"), gate.get("away_column")])
        req_cols = [c for c in req_cols if c]
        
        if all(c in df.columns for c in req_cols):
            valid_rules.append(rule)
        else:
            missing = [c for c in req_cols if c not in df.columns]
            print(f"[INFO] Pomijam regułę '{rule.get('name')}' w OOF (brak kolumn w SkillCorner: {missing})")

    oof_evidence_cfg = dict(evidence_cfg)
    oof_evidence_cfg["rules"] = valid_rules

    # --- 4. APLIKACJA NOWYCH REGUŁ EVIDENCE FUSION NA SUROWE HGB ---
    p_home_adj, p_away_adj, _, _ = apply_directional_evidence(
        df=df,
        p_home_model=df["hist_gradient_boosting_p_home"],
        evidence_config=oof_evidence_cfg,
        model_feature_names=[]
    )

    df["p_home_fused"] = p_home_adj
    df["p_away_fused"] = p_away_adj
    
    # Tworzymy bezpieczną kolumnę p_dead dla OOF (zakładamy 0.0 dla klatek Live lub 1.0 jeśli target==2)
    df["p_dead_oof"] = np.where(df["target"] == 2, 0.99, 0.01)

    p_home_col = "p_home_fused"
    p_away_col = "p_away_fused"
    p_dead_col = "p_dead_oof"

    # --- 4. WYKLUCZENIE MECZU TESTOWEGO I PODZIAŁ NA TRAIN/VAL ---
    TEST_MATCHES = ["1874553"]
    all_matches = sorted(df["match_id"].astype(str).unique())
    available_train_matches = [m for m in all_matches if m not in TEST_MATCHES]

    rng = np.random.default_rng(seed=42)
    train_mids = rng.choice(available_train_matches, size=15, replace=False)
    val_mids = [m for m in all_matches if m not in train_mids]
    
    all_needed_mids = list(train_mids) + list(val_mids)
    df = df[df['match_id'].astype(str).isin(all_needed_mids)].copy()

    print(f"Trening parametrów na: {len(train_mids)} meczach.")
    print(f"Weryfikacja parametrów (Val) na: {len(val_mids)} meczach (w tym chroniony: {TEST_MATCHES}).")

    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda t: objective(t, df, p_home_col, p_away_col, p_dead_col, train_mids, val_mids),
        n_trials=75,
        show_progress_bar=True
    )
    
    print("\n" + "="*50)
    print("OPTYMALNE PARAMETRY DLA OOF (SKALA 10 FPS):")
    best = study.best_params
    for k, v in best.items(): 
        print(f"  {k}: {v}")
        
    print(f"\n[BEST TRIAL] Train Score (15 meczów): {study.best_trial.user_attrs.get('train_score'):.4f}")
    print(f"[BEST TRIAL] Val Score (weryfikacja):   {study.best_trial.user_attrs.get('val_score'):.4f}")
    print("="*50)
    print("Proszę, wklej mi powyższe wyniki. Wykonam wtedy bezpieczną przesiadkę do main.py.")