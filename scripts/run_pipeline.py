from src.deadball import run_deadball_pipeline

# ... po zakończeniu Stage 1 (ROLA : tracking) ...

print(">>> [Stage 2] Segmentacja przerw w grze (Dead Ball)...")
stage2_df = run_deadball_pipeline(
    stage1_parquet_path="data/interim/stage1_tracking.parquet",
    output_parquet_path="data/interim/stage2_deadball.parquet",
)