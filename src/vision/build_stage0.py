# ===========================================================================
# MODULE 1: SETUP AND CONFIGURATION
# ===========================================================================
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

CURRENT_DIR = Path(__file__).resolve().parent  # src/vision
BASE_DIR = CURRENT_DIR.parent                  # src
MAIN_DIR = BASE_DIR.parent                     # folder główny

possible_paths = [
    MAIN_DIR / "games.py",
    MAIN_DIR / "starter" / "games.py",
    BASE_DIR / "starter" / "games.py"
]

GAMES_PY_PATH = next((p for p in possible_paths if p.exists()), None)
if not GAMES_PY_PATH:
    raise FileNotFoundError(f"Nie znaleziono pliku games.py! Przeszukano: {[str(p) for p in possible_paths]}")

spec = importlib.util.spec_from_file_location("games", str(GAMES_PY_PATH))
games_module = importlib.util.module_from_spec(spec)
sys.modules["games"] = games_module
spec.loader.exec_module(games_module)

SOURCES = games_module.SOURCES
SUFFIXES = games_module.SUFFIXES


# ===========================================================================
# MODULE 2: CAMERA LOOKAT COMPUTATION (INVERSE HOMOGRAPHY)
# ===========================================================================
def compute_broadcast_lookat(df_brd: pd.DataFrame, frame_w: int = 1920, frame_h: int = 1080):
    """
    Wylicza współrzędne boiska (cam_lookat_x, cam_lookat_y) w skali [0, 1],
    na które patrzy środek kamery Broadcast (piksel 960, 540).
    
    Zgodnie z DATA.md macierz H mapuje: boisko (0..1) -> piksele (1920x1080).
    Aby przejść z pikseli na boisko, używamy macierzy odwrotnej H^-1.
    """
    n_brd = len(df_brd)
    lookat_x = np.full(n_brd, np.nan, dtype=np.float64)
    lookat_y = np.full(n_brd, np.nan, dtype=np.float64)

    h_cols = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]
    if not all(c in df_brd.columns for c in h_cols):
        h_cols = [c for c in df_brd.columns if c.startswith("h") and len(c) <= 3]

    H_all = df_brd[h_cols].values.astype(np.float64).reshape(-1, 3, 3)
    valid_mask = (df_brd["valid"].values == 1) & (~np.isnan(H_all).any(axis=(1, 2)))

    # Odfiltrowanie macierzy osobliwych (wyznacznik bliski 0)
    dets = np.zeros(n_brd, dtype=np.float64)
    dets[valid_mask] = np.linalg.det(H_all[valid_mask])
    invertible_mask = valid_mask & (np.abs(dets) > 1e-12)

    valid_idx = np.where(invertible_mask)[0]
    if len(valid_idx) == 0:
        return lookat_x, lookat_y

    # Wektoryzowane odwrócenie wszystkich poprawnych macierzy 3x3 naraz
    H_inv = np.linalg.inv(H_all[valid_idx])

    # Środek klatki wideo w układzie jednorodnym: [960.0, 540.0, 1.0]
    center_px = np.array([frame_w / 2.0, frame_h / 2.0, 1.0], dtype=np.float64)

    # Mnożenie macierzowe: (N, 3, 3) @ (3,) -> (N, 3)
    proj = H_inv @ center_px
    w = proj[:, 2]

    good_w = np.abs(w) > 1e-9
    xs = np.where(good_w, proj[:, 0] / w, np.nan)
    ys = np.where(good_w, proj[:, 1] / w, np.nan)

    # Odrzucenie ekstremalnych błędów kalibracji (poza zakresem boiska z marginesem)
    in_pitch = (xs >= -0.25) & (xs <= 1.25) & (ys >= -0.25) & (ys <= 1.25)

    final_idx = valid_idx[in_pitch]
    lookat_x[final_idx] = xs[in_pitch]
    lookat_y[final_idx] = ys[in_pitch]

    return lookat_x, lookat_y


# ===========================================================================
# MODULE 3: TIMELINE ALIGNMENT AND PARQUET EXPORT
# ===========================================================================
def build_stage0_video():
    sync_json_path = MAIN_DIR / "data" / "interim" / "sync_offset.json"
    if not sync_json_path.exists():
        raise FileNotFoundError(
            f"Brak pliku {sync_json_path}! Uruchom najpierw sync_cameras.py."
        )

    with open(sync_json_path, "r", encoding="utf-8") as f:
        sync_data = json.load(f)

    brd_hom_path = MAIN_DIR / "data" / "broadcast" / (SOURCES["broadcast"]["stem"] + SUFFIXES["homography"])
    if not brd_hom_path.exists():
        brd_hom_path = list((MAIN_DIR / "data" / "broadcast").glob("*.csv"))[0]

    n_tac = SOURCES["tactical"]["n_video_frames"]    # 199,319
    n_brd = SOURCES["broadcast"]["n_video_frames"]   # 164,488

    print(f"[1/3] Wczytywanie homografii Broadcast: {brd_hom_path.name} ({n_brd} wierszy)...")
    df_brd = pd.read_csv(brd_hom_path)

    # Upewniamy się, że wiersze są posortowane po kolumnie frame (0 .. n_brd-1)
    col_f = "frame" if "frame" in df_brd.columns else df_brd.columns[0]
    df_brd = df_brd.sort_values(col_f).reset_index(drop=True)

    brd_valid = df_brd["valid"].fillna(0).values.astype(np.int8)
    brd_reproj = (
        df_brd["reprojection_error"].values.astype(np.float64)
        if "reprojection_error" in df_brd.columns
        else np.full(n_brd, np.nan)
    )

    print("[2/3] Wyliczanie punktu patrzenia kamery (cam_lookat_x, cam_lookat_y)...")
    brd_lookat_x, brd_lookat_y = compute_broadcast_lookat(df_brd)

    # Inicjalizacja docelowych kolumn na osi czasu Tactical (0 .. 199,318)
    tac_frames = np.arange(n_tac, dtype=np.int32)
    aligned_brd_frame = np.full(n_tac, -1, dtype=np.int32)
    aligned_valid = np.zeros(n_tac, dtype=np.int8)  # Domyślnie 0 (przerwa / poza meczem / powtórka)
    aligned_lookat_x = np.full(n_tac, np.nan, dtype=np.float64)
    aligned_lookat_y = np.full(n_tac, np.nan, dtype=np.float64)
    aligned_reproj = np.full(n_tac, np.nan, dtype=np.float64)

    print("[3/3] Przesuwanie osi czasu Broadcast na oś Tactical wg sync_offset.json...")
    for half_key, info in sync_data.items():
        tac_start = int(info["start_frame"])
        tac_end = int(info["end_frame"])
        offset = int(info["offset_frames"])  # tac_frame = brd_frame + offset

        # Wycinek klatek dla danej połowy na osi Tactical
        t_idx = np.arange(tac_start, tac_end + 1, dtype=np.int32)
        b_idx = t_idx - offset

        # Zabezpieczenie przed wyjściem poza zakres klatek wideo
        in_bounds = (t_idx >= 0) & (t_idx < n_tac) & (b_idx >= 0) & (b_idx < n_brd)
        t_sub = t_idx[in_bounds]
        b_sub = b_idx[in_bounds]

        aligned_brd_frame[t_sub] = b_sub
        aligned_valid[t_sub] = brd_valid[b_sub]
        aligned_lookat_x[t_sub] = brd_lookat_x[b_sub]
        aligned_lookat_y[t_sub] = brd_lookat_y[b_sub]
        aligned_reproj[t_sub] = brd_reproj[b_sub]

        valid_in_half = int(np.sum(aligned_valid[t_sub] == 1))
        print(
            f"   -> {half_key.upper()}: przesunięto o {offset:+d} kl. "
            f"(Tactical {tac_start}..{tac_end} <-> Broadcast {b_sub[0]}..{b_sub[-1]}) | "
            f"valid==1: {valid_in_half}/{len(t_sub)} kl."
        )

    # Budowa finalnej tabeli
    df_stage0 = pd.DataFrame({
        "frame": tac_frames,                  # Oś główna Tactical: 0 .. 199318
        "broadcast_frame": aligned_brd_frame, # Odpowiadająca klatka Broadcast (-1 poza połowami)
        "valid": aligned_valid,               # Dla Deadballa: 1 = kamera TV na boisku, 0 = powtórka/przerwa
        "cam_lookat_x": aligned_lookat_x,     # Dla Trackingu: współrzędna X środka kamery TV [0..1]
        "cam_lookat_y": aligned_lookat_y,     # Dla Trackingu: współrzędna Y środka kamery TV [0..1]
        "reprojection_error": aligned_reproj  # Błąd kalibracji homografii (im niższy, tym pewniejszy lookat)
    })

    out_dir = MAIN_DIR / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "stage0_video.parquet"

    df_stage0.to_parquet(parquet_path, index=False)

    total_valid = int(np.sum(aligned_valid == 1))
    total_lookat = int(np.sum(~np.isnan(aligned_lookat_x)))
    print("\n" + "=" * 72)
    print(f"[SUKCES] Zapisano plik: {parquet_path}")
    print(f"         Wiersze ogółem (oś Tactical): {len(df_stage0)}")
    print(f"         Klatki valid == 1 (gra live): {total_valid} ({100*total_valid/n_tac:.1f}%)")
    print(f"         Klatki valid == 0 (przerwy):  {n_tac - total_valid} ({100*(n_tac - total_valid)/n_tac:.1f}%)")
    print(f"         Wyliczone punkty cam_lookat:  {total_lookat}")
    print("=" * 72)

    return df_stage0


if __name__ == "__main__":
    build_stage0_video()