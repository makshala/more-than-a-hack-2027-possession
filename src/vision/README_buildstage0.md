# Vision & Alignment Pipeline

This module is responsible for the unsupervised time synchronization between cameras (Broadcast and Tactical) and translating homography matrices into physical pitch coordinates. The entire process is automated and relies on deterministic signal processing.

## Execution Order (Pipeline)

The scripts must be run in a strictly defined order, as they exchange generated data with one another:

### 1. `sync_cameras.py`
* **Purpose:** Finding the exact frame shift (offset) between the TV broadcast camera and the tactical camera.
* **How it works:** 
  * Uses OCR (`easyocr`) to read the match clock and establish a coarse offset.
  * Divides each half into sectors and uses a sliding window to overlay ball trajectories from `.atd` files.
  * Selects the offset that minimizes the median distance error and filters out anomalies using MAD statistics.
* **Output file:** `data/interim/sync_offset.json`

### 2. `build_stage0_video.py`
* **Purpose:** Building the base feature table for the classifier.
* **How it works:** Loads the previously calculated `sync_offset.json`. Then, for each frame, it inverts the homography matrix ($H^{-1}$) and projects the center of the Full HD screen (960, 540) onto the pitch.
* **Output file:** `data/interim/stage0_video.parquet` (contains `cam_lookat_x`, `cam_lookat_y`, `cam_fov_width`, `cam_speed`, among others).

* **Geometry and constants:** All parameters used in the code are physical constants (1920x1080 resolution, 25 fps framerate, mathematical pitch boundaries [0..1]) or standard statistical thresholds (MAD threshold).

## Architecture Decisions: Accepted vs. Rejected Approaches

During the development of the synchronization module, we explored different methods to align the timelines. Below is a summary of our design choices and why the final approach was selected.

### ACCEPTED: OCR Coarse Sync + ATD Tracking Refinement
**How it works:** 
1. **Coarse offset:** We use OCR (`easyocr`) on the broadcast match clock to establish a base frame shift.
2. **Fine offset:** We divide the timeline into sectors and use a sliding window over the provided `.atd` ball tracking coordinates to find the exact frame where trajectories perfectly match, filtering outliers with MAD.
**Why we chose this:**
* **Best of both worlds:** OCR instantly narrows down the search space to a few seconds, while the physical tracking data (`.atd`) provides frame-perfect precision.
* **Lightweight & Fast:** By limiting the search window with OCR, the algorithm avoids heavy global calculations.
* **Robustness:** It relies on the actual physical movement of the ball, completely bypassing visual noise (shadows, lighting, motion blur) that ruins image-based alignment.

### REJECTED: Visual Homography Warping (MSE)
**How it worked:** We attempted to warp the broadcast frame into the tactical perspective using homography matrices. The algorithm extracted green pitch masks and player silhouettes (Sobel gradients) and scanned a sliding window to minimize the Mean Squared Error (MSE) of overlapping pixels.
**Why it was rejected:**
* **Computationally Expensive:** Extracting frames via `cv2` and calculating pixel-perfect perspective warps per frame was extremely slow and resource-heavy.
* **Brittle to Visual Artifacts:** Minimizing MSE on player silhouettes is highly sensitive to lighting changes (harsh sun vs. shadows) and motion blur.
* **Homography Dependency:** The fine-tuning process heavily relied on the provided homography matrices being perfectly stable. Any jitter in the calibration caused the MSE alignment to fail.
* **Perspective Distortion of Players:** Homography matrices only correctly map a flat 2D plane (the pitch). When warping the low-angle broadcast view to match the high-angle tactical view, 3D objects like players' bodies become severely stretched, skewed, or flattened. This fundamental geometry issue makes comparing their silhouettes across cameras highly inaccurate.