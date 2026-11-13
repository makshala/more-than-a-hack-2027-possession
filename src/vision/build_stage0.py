# ===========================================================================
# MODULE 1: SETUP AND CONFIGURATION
# ===========================================================================
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

CURRENT_DIR = Path(__file__).resolve().parent  
BASE_DIR = CURRENT_DIR.parent                  
MAIN_DIR = BASE_DIR.parent                     

possible_paths = [
    MAIN_DIR / "games.py",
    MAIN_DIR / "starter" / "games.py",
    BASE_DIR / "starter" / "games.py"
]

GAMES_PY_PATH = next((p for p in possible_paths if p.exists()), None)
if not GAMES_PY_PATH:
    raise FileNotFoundError(f"games.py file not found! Searched in: {[str(p) for p in possible_paths]}")

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
    Computes pitch coordinates (cam_lookat_x, cam_lookat_y) in the [0, 1] scale,
    representing the point where the Broadcast camera center (pixel 960, 540) is looking.
    
    According to DATA.md, matrix H maps: pitch (0..1) -> pixels (1920x1080).
    To project from pixels to the pitch, we use the inverse matrix H^-1.
    """
    n_brd = len(df_brd)
    lookat_x = np.full(n_brd, np.nan, dtype=np.float64)
    lookat_y = np.full(n_brd, np.nan, dtype=np.float64)

    h_cols = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]
    if not all(c in df_brd.columns for c in h_cols):
        h_cols = [c for c in df_brd.columns if c.startswith("h") and len(c) <= 3]

    H_all = df_brd[h_cols].values.astype(np.float64).reshape(-1, 3, 3)
    valid_mask = (df_brd["valid"].values == 1) & (~np.isnan(H_all).any(axis=(1, 2)))

  
    dets = np.zeros(n_brd, dtype=np.float64)
    dets[valid_mask] = np.linalg.det(H_all[valid_mask])
    invertible_mask = valid_mask & (np.abs(dets) > 1e-12)

    valid_idx = np.where(invertible_mask)[0]
    if len(valid_idx) == 0:
        return lookat_x, lookat_y

    H_inv = np.linalg.inv(H_all[valid_idx])

    center_px = np.array([frame_w / 2.0, frame_h / 2.0, 1.0], dtype=np.float64)


    proj = H_inv @ center_px
    w = proj[:, 2]

    good_w = np.abs(w) > 1e-9
    xs = np.where(good_w, proj[:, 0] / w, np.nan)
    ys = np.where(good_w, proj[:, 1] / w, np.nan)

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
            f"Missing file {sync_json_path}! Run sync_cameras.py first."
        )

    with open(sync_json_path, "r", encoding="utf-8") as f:
        sync_data = json.load(f)

    brd_hom_path = MAIN_DIR / "data" / "broadcast" / (SOURCES["broadcast"]["stem"] + SUFFIXES["homography"])
    if not brd_hom_path.exists():
        brd_hom_path = list((MAIN_DIR / "data" / "broadcast").glob("*.csv"))[0]

    n_tac = SOURCES["tactical"]["n_video_frames"]    
    n_brd = SOURCES["broadcast"]["n_video_frames"]   

    print(f"[1/3] Loading Broadcast homography: {brd_hom_path.name} ({n_brd} rows)...")
    df_brd = pd.read_csv(brd_hom_path)

  
    col_f = "frame" if "frame" in df_brd.columns else df_brd.columns[0]
    df_brd = df_brd.sort_values(col_f).reset_index(drop=True)

    brd_valid = df_brd["valid"].fillna(0).values.astype(np.int8)
    brd_reproj = (
        df_brd["reprojection_error"].values.astype(np.float64)
        if "reprojection_error" in df_brd.columns
        else np.full(n_brd, np.nan)
    )

    print("[2/3] Computing camera look-at points (cam_lookat_x, cam_lookat_y)...")
    brd_lookat_x, brd_lookat_y = compute_broadcast_lookat(df_brd)

    tac_frames = np.arange(n_tac, dtype=np.int32)
    aligned_brd_frame = np.full(n_tac, -1, dtype=np.int32)
    aligned_valid = np.zeros(n_tac, dtype=np.int8)  
    aligned_lookat_x = np.full(n_tac, np.nan, dtype=np.float64)
    aligned_lookat_y = np.full(n_tac, np.nan, dtype=np.float64)
    aligned_reproj = np.full(n_tac, np.nan, dtype=np.float64)

    print("[3/3] Aligning Broadcast timeline to Tactical timeline using sync_offset.json...")
    for half_key, info in sync_data.items():
        tac_start = int(info["start_frame"])
        tac_end = int(info["end_frame"])
        offset = int(info["offset_frames"])  

      
        t_idx = np.arange(tac_start, tac_end + 1, dtype=np.int32)
        b_idx = t_idx - offset

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
            f"   -> {half_key.upper()}: shifted by {offset:+d} frames "
            f"(Tactical {tac_start}..{tac_end} <-> Broadcast {b_sub[0]}..{b_sub[-1]}) | "
            f"valid==1: {valid_in_half}/{len(t_sub)} frames"
        )

  
    df_stage0 = pd.DataFrame({
        "frame": tac_frames,                  
        "broadcast_frame": aligned_brd_frame, 
        "valid": aligned_valid,              
        "cam_lookat_x": aligned_lookat_x,     
        "cam_lookat_y": aligned_lookat_y,     
        "reprojection_error": aligned_reproj  
    })

    out_dir = MAIN_DIR / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "stage0_video.parquet"

    df_stage0.to_parquet(parquet_path, index=False)

    total_valid = int(np.sum(aligned_valid == 1))
    total_lookat = int(np.sum(~np.isnan(aligned_lookat_x)))
    print("\n" + "=" * 72)
    print(f"[SUCCESS] Saved file: {parquet_path}")
    print(f"          Total rows (Tactical timeline): {len(df_stage0)}")
    print(f"          Frames valid == 1 (live play):  {total_valid} ({100*total_valid/n_tac:.1f}%)")
    print(f"          Frames valid == 0 (breaks):     {n_tac - total_valid} ({100*(n_tac - total_valid)/n_tac:.1f}%)")
    print(f"          Computed cam_lookat points:     {total_lookat}")
    print("=" * 72)

    return df_stage0


if __name__ == "__main__":
    build_stage0_video()