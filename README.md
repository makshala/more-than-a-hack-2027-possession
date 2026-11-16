<div align="center">

# ⚽ Tactical Possession & Dead-Ball Estimation Engine
### **FC Barcelona x Metrica Sports — More than a Hack 2027**
**Team: IAD AGH** *(AGH University of Krakow — Data Science & Engineering)*

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Data Policy](https://img.shields.io/badge/Data-Proprietary%20%2F%20NDA%20Protected-informational)](#-data-privacy--execution-notice)

*A high-precision, low-latency spatio-temporal inference pipeline for frame-level and spell-level football possession decoding (`25 FPS`) combining multi-camera tracking, kinematic physics, Out-of-Fold Gradient Boosting, and explicit-duration Viterbi state machines.*

</div>

---

## 📖 Overview

In modern football analytics, evaluating possession solely at the individual frame level fails to capture tactical continuity. A jittery classifier can achieve high per-frame accuracy while shattering a single build-up phase into dozens of unusable micro-fragments.

This repository showcases **Team IAD AGH's** end-to-end engineering architecture developed for the **FC Barcelona x Metrica Sports (More than a Hack 2027)** challenge. Our system transforms noisy multi-source tracking and broadcast telemetry (`199,319 frames` at `25 FPS`) into clean, coach-ready tactical possession spells across four mutually exclusive states:
* **`H` (Home Possession)** — FC Barcelona actively controlling the ball in live play.
* **`A` (Away Possession)** — Athletic Club actively controlling the ball in live play.
* **`D` (Dead Ball / Stoppage)** — Ball out of bounds, fouls, set-piece preparation, post-goal intervals, and half-time.
* **`X` (Ungraded / Abstention)** — Pre-match, half-time break, and post-match ungraded video intervals.

### Key Engineering Highlights
* **Cross-FPS Phase-Shift Compensation & Asymmetric Smoothing (`lat_f = +0.8 frames`):** Compensates for the sampling phase shift between external SkillCorner training data (`10 FPS`) and target match telemetry (`25 FPS`, `shift = 3 frames`), paired with asymmetric zero-phase smoothing (`window = 37` for live `H/A` play while preserving sharp stoppages `D`) rather than noisy `leadlag` heuristics.
* **Physics-Informed Dead-Ball Detection:** Integrates FIFA opponent exclusion-zone geometry, ball-out-of-bounds kinematics, broadcast replay cues, and boundary impulse fluxes.
* **Domain-Invariant ML + Bounded Log-Odds Evidence Fusion:** Combines a `HistGradientBoostingClassifier` trained strictly on external SkillCorner Open Data (`20 matches`, `5-fold GroupKFold`) with deterministic physical ball-contact evidence in log-odds space.
* **Spell-Aware Temporal Decoding:** Eliminates sub-second duel flickers (`H -> A -> H` sandwiches) while strictly preserving genuine stoppages (`H -> D -> H`) that separate consecutive possessions.

---

## 👥 Team IAD AGH (Authors & Roles)

Our engineering workflow is divided into **5 specialized roles**, each responsible for a dedicated stage of the spatio-temporal pipeline:

| Role | Team Member | Responsibility & Module Scope | Links |
| :--- | :--- | :--- | :--- |
| **Role 1** | **Piotr Pęczek** | **Data Synchronization, Broadcast Context & Pipeline Orchestration**<br>Video-to-tracking temporal alignment (`sync_offset.json`), broadcast camera/replay extraction (`src/vision/`), and end-to-end orchestration. | [GitHub](https://github.com/) · [LinkedIn](https://linkedin.com/) |
| **Role 2** | **Maksymilian Hala** | **Spatio-Temporal Tracking & Kinematic Feature Engineering**<br>Savitzky-Golay kinematics, pitch control potential fields, Time-to-Intercept (`TTI`), and the 111-column feature contract (`src/tracking/build_stage1.py`). | [GitHub](https://github.com/makshala) · [LinkedIn](https://www.linkedin.com/in/maksymilian-hala-69232a316/) |
| **Role 3** | **Franciszek Sobczyk** | **Dead-Ball (`P(D)`) Estimation & Set-Piece Geometry**<br>SportsCode XML smart-tag priors, FIFA exclusion-zone physics, dual-edge impulse snapping, and 2-state Viterbi `P(D)` decoding (`src/deadball/`). | [GitHub](https://github.com/) · [LinkedIn](https://linkedin.com/) |
| **Role 4** | **Jan Potaczek** | **Live Possession ML (`P(H)`, `P(A)`) & Log-Odds Evidence Fusion**<br>SkillCorner V3 domain adaptation, `GroupKFold` HGB training, bounded directional log-odds fusion, and 3-state simplex calibration (`src/possession/`). | [GitHub](https://github.com/) · [LinkedIn](https://linkedin.com/) |
| **Role 5** | **Paweł Ryżewski** | **Temporal Spell Decoding, Phase-Shift Compensation & Evaluation**<br>Asymmetric multi-scale Viterbi decoding, cross-FPS phase alignment (`shift = 3`), spell-aware morphological cleanup, and submission export (`src/decode/`). | [GitHub](https://github.com/) · [LinkedIn](https://linkedin.com/) |

---

## 🏗️ System Architecture & 5-Role Pipeline

### How Each Stage Works
1. **Role 1 (Vision & Camera Synchronization — `src/vision/`):** Synchronizes tactical and broadcast video feeds with optical tracking (`sync_cameras.py`) and builds the aligned Stage 0 telemetry representation (`build_stage0.py`).
2. **Role 2 (Spatio-Temporal Feature Engineering — `src/tracking/`):** Computes 111 physical, spatial, and tactical features (`build_stage1.py`), including ball kinematics, player-to-ball distances, pitch control, goalkeeper context, and possession control windows.
3. **Role 3 (Dead-Ball Engine — `src/deadball/`):** Fuses SportsCode XML set-piece priors with pitch boundary geometry, FIFA exclusion-zone rules, and restart kick impulses inside a 2-state Viterbi decoder to output calibrated out-of-play probabilities `P(D)`.
4. **Role 4 (Live Possession ML & Evidence Fusion — `src/possession/`):** Predicts conditional live-play probabilities `P(H|live)` and `P(A|live)` using a `HistGradientBoostingClassifier` trained on 37 shared features from external SkillCorner Open Data, then refines predictions via bounded log-odds evidence rules (with noisy `leadlag` votes disabled) before fusing with Role 3's `P(D)`.
5. **Role 5 (Temporal Spell Decoder — `src/decode/`):** Decodes the 3-state probability simplex into contiguous possession spells using cross-FPS phase-shift compensation (`10 FPS -> 25 FPS`, `shift = 3 frames`), asymmetric smoothing (`window = 37` on `H/A` live play alongside a sharp `D` profile), calibrated log-Viterbi transition penalties, and spell-aware sandwich filtering.

---

## 🔒 Data Privacy & Execution Notice

> **Important Notice Regarding Reproducibility & Proprietary Data:**
> The competition dataset (including optical tracking telemetry, broadcast video feeds, tactical SportsCode XML annotations, and intermediate feature Parquets from the FC Barcelona vs. Athletic Club match) is proprietary property of **FC Barcelona** and **Metrica Sports**. Under strict competition rules and licensing agreements, **none of the match data files are included in this repository**.
>
> Consequently, **the full match inference pipeline cannot be executed out-of-the-box by external users** without access to the private competition dataset. The codebase is published as an open-source engineering showcase of our architecture, algorithms, and modelling methodology.

### Environment Setup & Synthetic Unit Tests (Runnable Without Proprietary Data)
While the match pipeline requires protected telemetry, you can set up the environment using **[uv](https://github.com/astral-sh/uv)** (requires **Python `>=3.11`**) and run our **synthetic Markovian possession generator and unit test suite** (`src/decode/test_decode.py`), which simulates realistic tracking occlusions, player scrambles, and broadcast camera cuts:

```bash
# 1. Clone the repository and sync the environment
git clone https://github.com/<your-org>/more-than-a-hack-2027-possession.git
cd more-than-a-hack-2027-possession
uv sync

# 2. Run the Role 5 temporal decoding test suite on synthetic telemetry
PYTHONPATH="src/decode:." uv run python src/decode/test_decode.py
```

### Reference Production Pipeline Execution (For Authorized Evaluators with Local Data)
When the licensed competition files are mounted locally inside `data/`, the complete 6-step pipeline (Roles 1 to 5 + Evaluation) is executed via `.\run_pipeline.bat` or step-by-step:

```bash
# Step 1: Role 1 — Camera Synchronization & Stage 0 Telemetry Build
uv run python src/vision/sync_cameras.py
uv run python src/vision/build_stage0.py

# Step 2: Role 2 — 111-Column Spatio-Temporal Feature Engineering (Stage 1)
uv run python src/tracking/build_stage1.py

# Step 3: Role 3 — Dead-Ball Probability Estimation (Stage 2)
uv run python src/deadball/d_model.py \
  --stage1 data/interim/stage1_tracking.parquet \
  --xml data/tactical/BAR-ATH_tactical_smart_tagging.xml \
  --output data/interim/stage2_deadball.parquet

# Step 4: Role 4 — HGB Live Possession Inference & 3-State Simplex Fusion (Stage 3)
uv run python src/possession/inference/run_role4_pipeline.py \
  --competition-parquet data/interim/stage1_tracking.parquet \
  --role3-parquet data/interim/stage2_deadball.parquet \
  --model role4_artifacts/role4_hist_gradient_boosting.joblib \
  --config src/possession/config.py \
  --output data/interim/stage3_role4_probs.parquet \
  --full-enriched-output data/interim/stage3_full_enriched.parquet \
  --role3-probability-column p_dead \
  --confirm-role3-out-of-play-only \
  --confirm-role4-conditional-live \
  --overwrite

# Step 5: Role 5 — Spell-Optimal Viterbi Decoding & Submission Export (Stage 5)
uv run python src/decode/main.py \
  --parquet data/interim/stage3_role4_probs.parquet \
  --sync data/interim/sync_offset.json \
  --out-parquet data/interim/stage5_decoded_spells.parquet \
  --out-csv data/interim/stage5_decoded_spells.csv \
  --out-submission "evaluation/results/IAD AGH.csv"

# Step 6: Submission Validation & Official Evaluation
uv run python evaluation/validate_submission.py "evaluation/results/IAD AGH.csv"
uv run python evaluation/run_evaluation.py
```
---

## 🛡️ Competition Compliance & Methodology

Our pipeline strictly adheres to all official rules of **More than a Hack 2027**:
1. **Zero Target-Match Leakage:** No machine learning weights, probability calibrators, or thresholds are fitted on the target match video or its ground-truth labels.
2. **External-Only Supervised Learning:** The Role 4 classifier consumes exclusively the 37 kinematic and spatial features shared with external SkillCorner Open Data, validated via 5-fold match-disjoint `GroupKFold`.
3. **Strict Data Confidentiality:** All proprietary match data, video files, and derived tracking tables are excluded from version control via `.gitignore`.

---

<div align="center">
  <sub>Built with ❤️ by <b>Team IAD AGH</b> at AGH University of Krakow for FC Barcelona x Metrica Sports (2026/2027).</sub>
</div>
