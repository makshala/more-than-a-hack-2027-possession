from src.decode.main import _smooth_1d_reflect, _clean_segments_spell_aware

import pandas as pd
import numpy as np
from pathlib import Path

# 1. Konfiguracja
match_id = "1874553"  
oof_path = "role4_artifacts/role4_oof_predictions.csv" # dopasuj ścieżkę, jeśli jest inna

# 2. Wczytanie predykcji modelu (Etap 4)
oof_df = pd.read_csv(oof_path)
match_oof = oof_df[oof_df["match_id"].astype(str) == str(match_id)].copy()

# Znajdujemy oś czasu
max_frame = int(match_oof["frame"].max())
all_frames = np.arange(max_frame + 1, dtype=np.int32)
known_frames = match_oof["frame"].to_numpy(dtype=np.int32)

# 3. Przygotowanie prawdopodobieństw dla Viterbiego (Etap 5)
# Domyślnie wszystko, czego nie było w Parquecie to Deadball (D=1.0)
prob_H = np.zeros(max_frame + 1)
prob_A = np.zeros(max_frame + 1)
prob_D = np.ones(max_frame + 1) # Luki to w 100% martwa piłka

# Podstawiamy prawdopodobieństwa z Twojego modelu HGB dla znanych klatek
# (Sprawdź jak dokładnie nazywają się te kolumny w Twoim pliku OOF, np. prob_0 i prob_1)
prob_H[known_frames] = match_oof["hist_gradient_boosting_p_home"].to_numpy()
prob_A[known_frames] = match_oof["hist_gradient_boosting_p_away"].to_numpy()
prob_D[known_frames] = 0.0  # Dla znanych klatek model dzieli 100% między H i A

# ==========================================
# 1. Time-Shifting dla skali 10 FPS
shift = 0
if shift > 0:
    p_h = np.pad(prob_H[shift:], (0, shift), mode='edge')
    p_a = np.pad(prob_A[shift:], (0, shift), mode='edge')
    p_d = np.pad(prob_D[shift:], (0, shift), mode='edge')
else:
    p_h, p_a, p_d = prob_H, prob_A, prob_D

# 2. Wygładzanie dla 10 FPS
p_h_smooth = _smooth_1d_reflect(p_h, window=5)
p_a_smooth = _smooth_1d_reflect(p_a, window=5)
p_d_smooth = _smooth_1d_reflect(p_d, window=4)

probs = np.stack([p_h_smooth, p_a_smooth, p_d_smooth], axis=1)
probs = np.clip(probs, 0.02, 0.98)
probs /= np.sum(probs, axis=1, keepdims=True)
log_emit = np.log(probs)

n = len(log_emit)

# 3. Kary z Optuny
pen_ha = 2.0
pen_hd = 2.2

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

# 4. Czyszczenie spelli dla 10 FPS
states_clean = _clean_segments_spell_aware(
    states=states,
    probs=probs,
    min_live_frames=7,
    min_dead_frames=10,
    live_sandwich_frames=11,
)

label_map = {0: "H", 1: "A", 2: "D"}
final_labels = [label_map[p] for p in states_clean]
# ==========================================

# 4. Zapis do formatu submission
sub_df = pd.DataFrame({
    "frame": all_frames,
    "label": final_labels
})

out_dir = Path("external results")
out_dir.mkdir(exist_ok=True)

# Czyszczenie starych wyników (żeby ewaluator się nie zgubił)
for old_file in out_dir.glob("*.csv"):
    old_file.unlink()

out_path = out_dir / f"{match_id}_submission.csv"
sub_df.to_csv(out_path, index=False)
print(f"Zapisano predykcje do {out_path}. Możesz odpalać ewaluator!")