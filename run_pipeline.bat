@echo off
set "TEAM_NAME=IAD AGH"
set "PYTHONPATH=evaluation;."
echo --- [1/6] Running Camera Synchronization ---
uv run python src/vision/sync_cameras.py

echo --- [2/6] Building Stage 0 (Broadcast Fusion) ---
uv run python src/vision/build_stage0.py

echo --- [3/6] Building Stage 1 (111-Column Tracking Contract) ---
uv run python src/tracking/build_stage1.py

echo --- [4/6] Running Role 3 (Dead-Ball Model) ---
uv run python src/deadball/d_model.py --stage1 data/interim/stage1_tracking.parquet --xml data/tactical/BAR-ATH_tactical_smart_tagging.xml --output data/interim/stage2_deadball.parquet

echo --- [5/6] Running Role 4 (ML Inference ^& Evidence Fusion) ---
uv run python src/possession/inference/run_role4_pipeline.py --competition-parquet data/interim/stage1_tracking.parquet --role3-parquet data/interim/stage2_deadball.parquet --model role4_artifacts/role4_hist_gradient_boosting.joblib --config src/possession/config.py --output data/interim/stage3_role4_probs.parquet --full-enriched-output data/interim/stage3_full_enriched.parquet --role3-probability-column p_dead --confirm-role3-out-of-play-only --confirm-role4-conditional-live --overwrite

echo --- [6/6] Running Role 5 (Temporal Decoding ^& Submission Export) ---
uv run python src/decode/main.py --parquet data/interim/stage3_role4_probs.parquet --sync data/interim/sync_offset.json --out-parquet data/interim/stage5_decoded_spells.parquet --out-csv data/interim/stage5_decoded_spells.csv --out-submission "evaluation/results/%TEAM_NAME%.csv"

echo --- [CHECK] Validating Official Submission File ---
uv run python evaluation/validate_submission.py "evaluation/results/%TEAM_NAME%.csv"

echo --- [SCORE] Running Public 5-Min Sanity Evaluation ---
uv run python evaluation/run_evaluation.py

echo --- PIPELINE FINISHED SUCCESSFULLY ---
pause