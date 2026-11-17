import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_parquet("src/possession/training/skillcorner_training.parquet")

gt_dir = Path("gt/skillcorner")
gt_dir.mkdir(parents=True, exist_ok=True)

for match_id, group in df.groupby("match_id"):
    max_frame = int(group["frame"].max())
    
    all_frames = np.arange(max_frame + 1, dtype=np.int32)
    
    labels = np.full(max_frame + 1, "D", dtype=object)
    
    known_frames = group["frame"].to_numpy(dtype=np.int32)
    known_targets = group["target"].to_numpy()
    
    labels[known_frames[known_targets == 0]] = "H"
    labels[known_frames[known_targets == 1]] = "A"
    
    gt_df = pd.DataFrame({
        "frame": all_frames,
        "label": labels
    })
    
    out_path = gt_dir / f"{match_id}_possession_gt.csv"
    gt_df.to_csv(out_path, index=False)
    print(f"Generated continuous GT for match: {match_id} (frames: 0 - {max_frame})")