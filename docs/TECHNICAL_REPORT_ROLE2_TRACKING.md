# Technical Documentation — Stage 1: Multi-Sensor Tracking, Physics & Tactical Feature Engineering

## 1. The Approach (Tracking & Feature Engineering Architecture)
Rather than treating automatic tracking data (ATD) as clean coordinates, our Stage 1 pipeline (`src/tracking/`) models the pitch as a noisy, multi-sensor physical dynamical system. Because the competition ranking metric (`spellF1`) heavily penalizes possession fragmentation (false transitions during ball flight or sensor dropouts) and boundary drift across temporal IoU thresholds [0.3 .. 0.7], our tracking architecture is built around three core pillars:
1. **Multi-Camera Sensor Fusion (Tactical + Broadcast):** Bridging the 40.7% missing ball rate in graded tactical frames by projecting synchronized Broadcast ATD ball detections and the TV camera optical axis (`LookAt`) onto the tactical timeline.
2. **Uncertainty-Aware Kinematics:** Replacing naive linear interpolation over long occlusions with a hierarchical Focal Action Centroid and explicitly quantifying frame-level positional uncertainty (`ball_sigma_m`) for downstream Quality Gating.
3. **Algorithmic Possession Proxies:** Translating the semantic rules of CHALLENGE.md into deterministic physical state machines and continuous spatial control fields using pure physics and geometry.

The entire Stage 1 pipeline is 100% deterministic, fully vectorized in NumPy/SciPy, executes in ~11 seconds for all 199,319 frames, and produces a strictly audited, zero-NaN data contract (`data/interim/stage1_tracking.parquet`, 111 columns).

---

## 2. Assets Used, Assets Dropped, and Rejected Approaches

### 2.1. Assets Used and How
We integrated all provided modalities across both camera feeds:
* **Tactical ATD (`BAR-ATH_tactical_atd.txt` & `.xml`):** Primary reference timeline (199,319 frames at 25 fps) and coordinate frame (105.3 x 68.0 m) for collective team geometry, player kinematics, and primary ball observations.
* **Synchronized Broadcast ATD & Homography (`stage0_video.parquet`):** Aligned to the tactical timeline via clock OCR and ATD ball trajectory matching (`sync_cameras.py`, `build_stage0.py`). Used for:
  1. **Direct Ball Gap Patching:** Injecting valid Broadcast ball coordinates (`brd_ball_x`, `brd_ball_y`) when the tactical ball is missing (`NaN`).
  2. **Camera Optical Axis (`brd_lookat_x`, `brd_lookat_y`, `brd_lookat_speed_ms`):** Computed via vectorized inverse homography of the broadcast frame center to guide latent ball estimation during long occlusions and detect camera whip-pans on clearances/restarts.
  3. **TV Replay & Cut Detector (`brd_is_replay`, `frames_since_brd_replay`):** Identifying graded frames where the match is ongoing but the broadcast view is invalid, providing a high-precision indicator of dead-ball stoppages (D).
* **Tactical Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`):** All 638 automatic phase and set-piece intervals are parsed onto the 25 fps timeline (`tactics.py`) to extract independent tactical priors and set-piece indicators.
* **Tactical Homography (`BAR-ATH_tactical_homography.csv`):** Used to compute reprojection error, camera pan speed, and project estimated ball coordinates back into image space.
* **Official Match Boundaries:** Graded half intervals (11,185–90,651 and 110,808–190,197) and the documented pre-kickoff dead-ball window at the start of the 2nd half (110,808–113,741).

### 2.2. Dropped Assets & Rejected Technical Approaches (And Why)
Following SUBMISSION.md guidelines, below are the modalities and algorithmic alternatives we tested and explicitly rejected:

| Rejected Approach / Asset | Reason for Rejection & Empirical Finding |
| :--- | :--- |
| **1. Broadcast Player Track Slots** | **Dropped.** While Broadcast ball coordinates are highly valuable, Broadcast player tracks suffer from severe camera zoom cropping and frequent timeline cuts. Fusing partial player sets introduced discontinuous jumps in team centroids and Pitch Control fields. Tactical ATD was retained as the sole source for player geometry. |
| **2. Pure Linear Interpolation Across All Gaps** | **Rejected.** Interpolating the ball across gaps larger than 1.0 second created unphysical "ghost trajectories" cutting straight across the pitch during out-of-play stoppages. Passing players accidentally crossed this artificial line, triggering false possession turnovers. We restricted linear interpolation strictly to gaps under 1 second and replaced long gaps with the Focal Action Centroid. |
| **3. Dynamic Linear Clock-Drift Warping** | **Evaluated & Simplified.** We implemented a linear regression drift model to test for quartz clock skew between the cameras. Empirical auditing revealed a cumulative drift of only 0.33 frames across the 1st half and 1.00 frame across the 2nd half. A static median offset per half achieves identical frame alignment without risking boundary rounding jitter. |
| **4. Naive Smart Tagging Consumption** | **Rejected.** Using the XML out-of-the-box introduced two critical errors: defensive tags label the defending team rather than the ball carrier, and 36 instances (including all Throw-ins) lack a team assignment. We implemented polarity inversion for defensive tags and a geometric touchline resolver for unassigned events. |
| **5. Finite-Difference Player Velocities** | **Rejected.** Raw frame-to-frame differences amplified tracker bounding-box jitter into velocity spikes over 20 m/s, corrupting Time-to-Intercept calculations. We replaced it with local Hungarian stitching, 7-frame Savitzky-Golay filtering, and permutation-invariant top-3 nearest set features. |

---

## 3. Training Data & Pre-trained Models
* **Stage 1 Tracking & Tactical Pipeline (`src/tracking/`):** Uses zero labelled training data (neither from the provided match nor from external football datasets). All feature transformations, tracklet stitching, filtering, and possession state machines are 100% deterministic, analytical algorithms grounded in pitch geometry and physics. No parameters were tuned or fitted against the public 5-minute ground truth or the match video.
* **Stage 0 Camera Synchronization (`src/vision/`):** Uses the open-source pre-trained EasyOCR (en) text-recognition model solely at inference time to read the scoreboard clock for coarse timeline alignment, refined deterministically via ATD ball trajectory median distance.

---

## 4. Handling Missing and Noisy Tracking Data

### 4.1. Player Tracklet Healing & De-duplication (`physics.py: heal_player_tracks`)
Raw ATD contains 46 anonymous slots with fragmentation and duplicate detections. Within each team:
* **Vectorized Hungarian Tracklet Stitching:** Tracklet terminations ("deaths") and appearances ("births") separated by 25 frames or less (1.0 second) are matched by minimizing the distance between the predicted trajectory and actual appearance (max distance 2.5 meters), linearly bridging short occlusion gaps.
* **Clone Removal:** Simultaneous same-team slots separated by less than 0.45 meters are resolved by retaining the tracklet with the higher 51-frame local temporal stability score.
* **Savitzky-Golay Kinematics:** Player positions and velocities are smoothed using a 7-frame, 2nd-order Savitzky-Golay convolutional filter, clamping unphysical velocity spikes above 12.5 m/s.

### 4.2. Three-Tier Ball Imputation & Uncertainty Quantification (`physics.py`)
After fusing Broadcast ball detections, every frame is assigned a status code and an explicit spatial uncertainty metric (sigma):
1. **OBSERVED (64.4% of graded frames):** Direct ATD measurement from Tactical (sigma = 0.5 m) or patched from synchronized Broadcast (sigma = 1.1 m).
2. **INTERPOLATED_SHORT (0.1% of graded frames):** Bridged via linear interpolation for gaps up to 1.0 second, with uncertainty scaling toward the gap midpoint (sigma between 0.66 and 2.5 m).
3. **INFERRED_LATENT (35.5% of graded frames):** For gaps longer than 1.0 second, computed via a two-stage Focal Action Centroid:
   * *Stage 1:* Speed-weighted player centroid blended 65%/35% with the Broadcast camera LookAt point.
   * *Stage 2:* Re-weighting all players by a spatial kernel and positive convergence velocity toward the Stage 1 center. Uncertainty scales dynamically with player dispersion (bounded between 2.5 and 25.0 meters).

---

## 5. Implemented Possession Rules & Domain Features (`tactics.py`)
To adhere strictly to the possession rules without using ground-truth labels, we engineered deterministic physical proxies:
* **Controlled Touch vs. Deflection State Machine:**
  * Calculates the incoming and outgoing ball trajectory deflection angle.
  * A valid team proximity frame requires the ball to be observed or shortly interpolated, the nearest player distance under 1.8 meters, the opponent over 2.0 meters away, and either sustained proximity (at least 4 out of 8 frames) or a low-angle controlled touch.
  * Distinct touch episodes (separated by more than 6 frames of ball flight) are counted per half. The last controlled team switches only on sustained control (at least 4 frames) or 2 or more consecutive touches by the same team, ignoring single-touch rebounds and deflections.
* **Kinematic Pitch Control & Time-to-Intercept:**
  * Computes each player's reaction-adjusted position (position + velocity * 0.4 seconds) and Time-to-Intercept assuming a max speed of 7.5 m/s.
  * Aggregates exponential arrival rates into continuous team spatial dominance (Pitch Control).
* **Multi-Scale & Edge-Transition Temporal Context:**
  * Computes both centered and causal rolling windows over 2-second and 4-second intervals per half, plus forward-backward Lead/Lag deltas to sharpen possession transition boundaries.
* **Canonical Half Normalization & Goalkeeper Proxies:**
  * Normalizes pitch coordinates per half so Home always attacks left-to-right, dynamically identifies the deepest player per team as Goalkeeper, and flags uncontested penalty-area ball retention.

---

## 6. Quantitative Final Effects (Stage 1 Pipeline Audit)
Executing `build_stage1.py` on the full match yields the following verified improvements over the raw dataset:

* **Direct Ball Visibility:** Recovered +8,094 genuine frames via Broadcast ATD fusion, raising coverage from 59.3% to 64.4% of graded frames.
* **High-Confidence Ball State:** Only 91 frames (0.1%) require short linear bridging, providing a highly reliable foundation for downstream models.
* **Long Ball Occlusions:** 100% of the 56,458 missing frames (35.5% of the graded match) are imputed via the Focal Action Centroid combined with Broadcast LookAt, accompanied by explicit uncertainty metrics.
* **Broadcast Replay Detection:** Pinpoints TV replays and close-ups during stoppages on 52,712 frames (33.2% of the graded match).
* **Smart Tagging Resolution:** 100% of Throw-ins and unassigned set pieces (36/36) assigned to a team via touchline geometry. Defensive polarity inverted on all defensive tags to eliminate systematic possession errors.
* **Data Completeness:** Outputs 199,319 rows across 111 columns with 0 missing values, executing end-to-end in approximately 11.0 seconds.

---

## 7. Limitations
1. **Simultaneous Long Occlusions in Both Feeds:** When the ball leaves the field or enters a dense penalty-box scramble during a TV replay, both cameras lose direct sight of the ball. While the Focal Action Centroid tracks the collective center of player convergence, exact micro-touches inside the crowd remain unobservable until the ball emerges.
2. **2D Homography Projection of Aerial Balls:** Both Tactical and Broadcast ATD project 3D lofted passes onto the 2D pitch plane. During high clearances, the apparent ground-plane coordinate shifts along the camera-to-ball ray, temporarily inflating 2D distances to players underneath the ball.
3. **Anonymous Slot Identity Switches:** Because ATD provides no player IDs, occlusions lasting longer than 1.0 second cannot be deterministically stitched without visual re-identification. Therefore, all team features are formulated as permutation-invariant set statistics.

---

## 8. Reproduction (Stage 0 & Stage 1 Pipeline)
From the repository root, with match assets unpacked in `data/`:

```bash
# 1. Build Stage 0 synchronized Broadcast features & ball coordinates:
uv run python src/vision/build_stage0.py

# 2. Build Stage 1 Tracking & Tactical contract (199,319 rows x 111 columns, ~11s runtime):
uv run python src/tracking/build_stage1.py