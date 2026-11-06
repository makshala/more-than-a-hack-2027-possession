# ===========================================================================
# MODULE 1: SETUP AND CONFIGURATION
# ===========================================================================
import cv2
import json
import re
import sys
import numpy as np
import pandas as pd
import easyocr
from pathlib import Path
import importlib.util

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
MAIN_DIR = BASE_DIR.parent

possible_paths = [
    MAIN_DIR / "starter" / "games.py",
    MAIN_DIR / "staarters" / "games.py",
    BASE_DIR / "starter" / "games.py"
]

GAMES_PY_PATH = next((p for p in possible_paths if p.exists()), None)
if not GAMES_PY_PATH:
    raise FileNotFoundError(f"games.py file not found! Searched in: {[str(p) for p in possible_paths]}")

spec = importlib.util.spec_from_file_location("games", str(GAMES_PY_PATH))
games_module = importlib.util.module_from_spec(spec)
sys.modules["games"] = games_module
spec.loader.exec_module(games_module)

half_video_blocks = games_module.half_video_blocks
SOURCES = games_module.SOURCES
SUFFIXES = games_module.SUFFIXES


# ===========================================================================
# MODULE 2: COARSE OFFSET VIA OCR
# ===========================================================================
def extract_time_in_seconds(ocr_results: list) -> int:
    pattern = re.compile(r'(\d{1,3})[\:\.\s\-;,\*]+(\d{2})')
    for text in ocr_results:
        match = pattern.search(text.strip())
        if match:
            m, s = int(match.group(1)), int(match.group(2))
            if 0 <= m <= 130 and 0 <= s < 60:
                return m * 60 + s
    return None

def get_coarse_offset(reader: easyocr.Reader, tac_vid: Path, brd_vid: Path,
                      start_frame: int, end_frame: int, valid_b: set, fps: float = 25.0) -> int:
    cap_b = cv2.VideoCapture(str(brd_vid))
    cap_t = cv2.VideoCapture(str(tac_vid))

    search_start = start_frame + int(60 * fps)
    step_frames = int(10 * fps)
    ocr_offsets = []

    print("   [OCR] Started scanning for the clock...")
    for f_idx in range(search_start, end_frame - step_frames, step_frames):
        if f_idx not in valid_b:
            continue

        cap_b.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        cap_t.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret_b, fb = cap_b.read()
        ret_t, ft = cap_t.read()
        if not (ret_b and ret_t):
            continue

        hb, ht = int(fb.shape[0] * 0.15), int(ft.shape[0] * 0.15)
        tb = extract_time_in_seconds(reader.readtext(fb[:hb, :], detail=0))
        tt = extract_time_in_seconds(reader.readtext(ft[:ht, :], detail=0))

        if tb is not None and tt is not None and abs(tb - tt) < 1800:
            coarse_cand = int(round((tb - tt) * fps))
            ocr_offsets.append(coarse_cand)
            print(f"   [OCR BASE] Reading #{len(ocr_offsets)} @ frame {f_idx} (BRD: {tb}s, TAC: {tt}s) -> {coarse_cand} frames")
            if len(ocr_offsets) >= 5:
                break

    cap_b.release()
    cap_t.release()

    if not ocr_offsets:
        raise RuntimeError(f"Failed to read OCR clock in the range ({start_frame} - {end_frame})!")

    final_coarse = int(round(np.median(ocr_offsets)))
    print(f"   [OCR BASE] Selected base offset: {final_coarse} frames")
    return final_coarse


# ===========================================================================
# MODULE 3: FINE ALIGNMENT VIA TRACKING DATA
# ===========================================================================
def load_atd_ball_and_players(atd_path: Path, n_frames: int):
    bx = np.full(n_frames, np.nan, dtype=np.float64)
    by = np.full(n_frames, np.nan, dtype=np.float64)

    with open(atd_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            idx = int(parts[0]) - 1
            if not (0 <= idx < n_frames):
                continue
            b_str = parts[2].strip()
            if b_str and not b_str.startswith("N"):
                vals = b_str.split(",")
                try:
                    bx[idx] = float(vals[0])
                    by[idx] = float(vals[1])
                except (ValueError, IndexError):
                    pass
    return bx, by

def refine_sector_by_atd_window(tac_bx: np.ndarray, tac_by: np.ndarray,
                                brd_bx: np.ndarray, brd_by: np.ndarray,
                                sec_start_brd: int, sec_end_brd: int,
                                valid_b_mask: np.ndarray,
                                coarse_offset: int, window: int = 30) -> tuple[int, int]:
    b_indices = np.arange(sec_start_brd, sec_end_brd)
    b_indices = b_indices[(b_indices >= 0) & (b_indices < len(brd_bx))]
    b_indices = b_indices[valid_b_mask[b_indices]]

    if len(b_indices) < 30:
        return None, None

    best_off = None
    min_err = float("inf")

    valid_ball_b = b_indices[~np.isnan(brd_bx[b_indices])]
    if len(valid_ball_b) < 20:
        return None, None

    anchor_brd = int(valid_ball_b[len(valid_ball_b) // 2])

    for lag in range(-window, window + 1):
        cand_off = coarse_offset + lag
        t_indices = b_indices + cand_off

        in_bounds = (t_indices >= 0) & (t_indices < len(tac_bx))
        b_sub = b_indices[in_bounds]
        t_sub = t_indices[in_bounds]

        both_vis = (~np.isnan(brd_bx[b_sub])) & (~np.isnan(tac_bx[t_sub]))
        if np.sum(both_vis) < 20:
            continue

        dx = tac_bx[t_sub[both_vis]] - brd_bx[b_sub[both_vis]]
        dy = tac_by[t_sub[both_vis]] - brd_by[b_sub[both_vis]]
        dist = np.sqrt(dx**2 + dy**2)

        err = float(np.median(dist))
        if err < min_err:
            min_err = err
            best_off = cand_off

    return best_off, anchor_brd


# ===========================================================================
# MODULE 4: OUTLIER FILTERING
# ===========================================================================
def filter_outliers_and_fit_drift(anchors: list, offsets: list, default_coarse: int):
    if not offsets:
        return default_coarse, 0.0, float(default_coarse), []

    arr_off = np.array(offsets, dtype=np.float64)
    arr_anc = np.array(anchors, dtype=np.float64)

    med = np.median(arr_off)
    mad = np.median(np.abs(arr_off - med))
    threshold = max(2.0, 2.5 * mad)

    inlier_mask = np.abs(arr_off - med) <= threshold
    clean_off = arr_off[inlier_mask]
    clean_anc = arr_anc[inlier_mask]

    final_median = int(round(np.median(clean_off)))

    if len(clean_off) >= 4:
        slope, intercept = np.polyfit(clean_anc, clean_off, 1)
    else:
        slope, intercept = 0.0, float(final_median)

    clean_pairs = [{"anchor_frame": int(a), "offset": int(o)} for a, o in zip(clean_anc, clean_off)]
    return final_median, float(slope), float(intercept), clean_pairs


# ===========================================================================
# MODULE 5: MAIN EXECUTION PIPELINE
# ===========================================================================
if __name__ == "__main__":
    tac_vid = list((MAIN_DIR / "data" / "tactical").glob("*.mp4"))[0]
    brd_vid = list((MAIN_DIR / "data" / "broadcast").glob("*.mp4"))[0]
    df_tac = pd.read_csv(list((MAIN_DIR / "data" / "tactical").glob("*.csv"))[0])
    df_brd = pd.read_csv(list((MAIN_DIR / "data" / "broadcast").glob("*.csv"))[0])

    tac_atd = MAIN_DIR / "data" / "tactical" / (SOURCES["tactical"]["stem"] + SUFFIXES["atd_raw"])
    brd_atd = MAIN_DIR / "data" / "broadcast" / (SOURCES["broadcast"]["stem"] + SUFFIXES["atd_raw"])

    n_tac = SOURCES["tactical"]["n_video_frames"]
    n_brd = SOURCES["broadcast"]["n_video_frames"]

    print("[INFO] Loading ATD points for precise alignment within the OCR window...")
    tac_bx, tac_by = load_atd_ball_and_players(tac_atd, n_tac)
    brd_bx, brd_by = load_atd_ball_and_players(brd_atd, n_brd)

    col_b = "frame" if "frame" in df_brd.columns else df_brd.columns[0]
    col_t = "frame" if "frame" in df_tac.columns else df_tac.columns[0]
    valid_b_set = set(df_brd[df_brd["valid"] == 1][col_b])
    valid_t_set = set(df_tac[df_tac["valid"] == 1][col_t])

    valid_b_mask = np.zeros(n_brd, dtype=bool)
    for f_id in valid_b_set:
        if 0 <= f_id < n_brd:
            valid_b_mask[f_id] = True

    max_frame_tac = int(df_tac[col_t].max())
    reader = easyocr.Reader(["en"], gpu=False)

    match_name = brd_vid.name.split("_")[0]
    blocks = half_video_blocks(match_name)
    halves = {}
    for half_num, start_f, end_f in blocks:
        halves[f"half_{half_num}"] = {"start": start_f, "end": end_f}

    print(f"\n[INFO] Loaded half boundaries for {match_name}: {halves}")

    results = {}
    num_samples = 20
    window_size = 30  

    for half_key, bounds in halves.items():
        print(f"\n=== HALF SYNCHRONIZATION ({num_samples} SECTORS): {half_key.upper()} ===")

        coarse = get_coarse_offset(
            reader, tac_vid, brd_vid,
            start_frame=bounds["start"],
            end_frame=bounds["end"],
            valid_b=valid_b_set
        )

        brd_half_start = max(0, bounds["start"] - coarse)
        brd_half_end = min(n_brd - 1, bounds["end"] - coarse)
        sector_size = (brd_half_end - brd_half_start) // num_samples

        sample_offsets = []
        sample_anchors = []

        for i in range(num_samples):
            sec_start = brd_half_start + i * sector_size
            sec_end = brd_half_start + (i + 1) * sector_size

            precise_off, used_anchor = refine_sector_by_atd_window(
                tac_bx, tac_by, brd_bx, brd_by,
                sec_start, sec_end, valid_b_mask,
                coarse_offset=coarse, window=window_size
            )

            if precise_off is not None:
                sample_offsets.append(precise_off)
                sample_anchors.append(used_anchor)
                print(f"   -> Sector {i+1:02d}/{num_samples} @ BRD {used_anchor}: offset = {precise_off} frames")
            else:
                print(f"   -> [INFO] Sector {i+1:02d}/{num_samples} skipped (missing continuous ball / break)")

        final_offset, slope, intercept, clean_pairs = filter_outliers_and_fit_drift(
            sample_anchors, sample_offsets, coarse
        )

        print(f"\n-> FINAL MEDIAN OFFSET ({half_key.upper()}): {final_offset} frames (from {len(clean_pairs)} clean sectors)")

        results[half_key] = {
            "start_frame": bounds["start"],
            "end_frame": bounds["end"],
            "offset_frames": final_offset,
            "coarse_offset": coarse,
            "drift_slope": slope,
            "drift_intercept": intercept,
            "sampled_offsets": sample_offsets,
            "clean_anchor_pairs": clean_pairs
        }

    out_dir = MAIN_DIR / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sync_offset.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\n[SUCCESS] Saved offsets for both halves to: {out_file}")