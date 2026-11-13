# Advanced Football Possession Temporal Decoding Ecosystem

A mathematically rigorous temporal decoding and spell extraction architecture for football possession phase estimation. Specifically designed to achieve maximum correctness, physical plausibility, and the highest possible **SpellF1** score by addressing the physical realities of tactical football.

---

## 1. Architectural Blueprint & Mathematical Foundations

```
Raw Upstream Posteriors P(S|X)
   │
   ▼
[Bayesian Inversion Layer] ───► True Log-Emissions: log P(X|S) = log P(S|X) - (1/τ) * log P(S)
   │                           (Strictly prevents prior double-counting & sequence lag)
   ▼
[Explicit Duration HSMM Trellis] ───► Continuous Duration Gamma Penalties log f(d; k, θ)
   │                                  Dynamic Context-Aware Transition Matrix A_t(i, j)
   ▼
Optimal State Sequence
   │
   ▼
[1D Temporal Morphological Filter] ───► Temporal Closing (bridges tracking micro-dropouts)
   │                                   Temporal Opening (erodes isolated single-frame spikes)
   ▼
[Temporal Hysteresis & Snapping] ───► Snaps boundaries to nearest stable tracking frames
   │
   ▼
Continuous Tactical Possession Spells (H, A, D) with tIoU & Penalized SpellF1 Evaluation
```

---

## 2. Core Modules

| Script | Purpose & Mathematical Enhancements |
| :--- | :--- |
| `config.py` | Defines football possession state spaces (`H`: Home, `A`: Away, `D`: Dead Ball), physical possession limits stored in **absolute milliseconds** (rather than frame counts) to maintain invariance across camera frame rates, strict logarithmic underflow boundaries (`LOG_EPS = 1e-15`), and hyperparameter search grids. |
| `generator.py` | Models empirically derived Dirichlet tracking error distributions and bursty Markovian noise regimes (`Clean`, `Occlusion`, `Player Clustering / Scramble`, `Broadcast Camera Cut`). Accurately replicates upstream computer vision failure modes. |
| `viterbi_decoder.py` | Dynamic context-aware Viterbi decoder. Employs real-time entropy-modulated transition matrices and executes the exact **Bayesian inversion sequence** to recover true emission log-likelihoods without double-counting priors. |
| `hsmm_decoder.py` | Explicit-duration Hidden Semi-Markov Model (HSMM). Replaces naive cutoffs with continuous parametric probability density functions (Gamma & Poisson) directly integrated into the dynamic programming trellis via $O(1)$ prefix sums. |
| `spells.py` | Advanced spell extraction integrating 1D mathematical morphology (closing & opening) and temporal hysteresis edge snapping. Aligns boundaries to points of high tracking confidence and maximum probability gradient. |
| `diagnostics.py` | Deprecates naive frame-level accuracy. Implements bipartite Hungarian matching over 1D Temporal Intersection over Union (tIoU) and computes an aggressive **fragmentation penalty** targeting spurious possession splits. |
| `main.py` | Comprehensive hyperparameter optimization engine. Autonomously navigates transition stickiness, turnover routing, duration weights, Gamma shapes, and morphological kernels to maximize the penalized SpellF1 score. |

---

## 3. Mathematical Formulations

### A. Bayesian Inversion Sequence
The upstream ML tracking model produces state posteriors $P(S_t = s \mid X_t)$. By Bayes' Rule:
$$P(X_t \mid S_t = s) = \frac{P(S_t = s \mid X_t) P(X_t)}{P(S_t = s)}$$
Operating in the log-domain with prior temperature $\tau$:
$$\log P(X_t \mid S_t = s) = \log P(S_t = s \mid X_t) - \frac{1}{\tau} \log P(S_t = s) + C_t$$
This eliminates prior double-counting, which is the leading cause of temporal boundary lag.

### B. Continuous Duration Penalties in HSMM Trellis
State durations follow a continuous Gamma distribution:
$$f(d; k, \theta) = \frac{1}{\Gamma(k)\theta^k} d^{k-1} e^{-d/\theta}$$
$$\log f(d; k, \theta) = (k - 1) \log d - \frac{d}{\theta} - \log \Gamma(k) - k \log \theta$$
When shape $k > 1$, as $d \to 0$, $\log f(d) \to -\infty$. This mathematically prevents impossible micro-possessions while rewarding sustained tactical phases.

Dynamic programming recurrence for segment $d$ ending at frame $t$:
$$V(t, s) = \max_{d \in [d_{\min}, d_{\max}]} \left[ \max_{s' \neq s} [V(t - d, s') + \log A(s', s)] + \omega_d \log f(d \mid s) + \sum_{\tau=t-d+1}^t \log P(X_\tau \mid S_\tau = s) \right]$$

### C. Temporal Intersection over Union (tIoU) & Fragmentation Penalty
For a ground truth spell $G = [s_g, e_g]$ and predicted spell $P = [s_p, e_p]$:
$$\text{tIoU}(G, P) = \frac{\max(0, \min(e_g, e_p) - \max(s_g, s_p) + 1)}{\max(e_g, e_p) - \min(s_g, s_p) + 1}$$

If a single ground truth spell is split into $K > 1$ predicted micro-spells, the excess fragments $E = \sum_m \max(0, K_m - 1)$ incur a geometric penalty:
$$\Phi_{\text{frag}} = \rho^{\sum_m (K_m - 1)} \quad (\rho = 0.75)$$
$$\text{SpellF1}_{\text{penalized}} = \text{SpellF1}(\text{tIoU} \ge 0.5) \times \Phi_{\text{frag}}$$

---

## 4. Benchmark Performance Matrix

Evaluation across 6,777 match frames (~271 seconds) under bursty occlusions, clustering scrambles, and camera cuts:

| Decoder Architecture | Penalized SpellF1 | Raw SpellF1 (@ tIoU=0.5) | Precision | Recall | Mean tIoU | Excess Frags | Spells Count | Frame Acc (Deprecated) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Naive Argmax Baseline** | **0.00%** | 0.36% | 0.18% | 11.11% | 0.579 | 145 | 1082 | 84.24% |
| **2. Static HMM Viterbi** | **0.00%** | 22.43% | 13.48% | 66.67% | 0.816 | 31 | 89 | 91.56% |
| **3. Dynamic Context Viterbi** | **0.00%** | 23.85% | 14.29% | 72.22% | 0.882 | 34 | 91 | 92.70% |
| **4. Baseline Explicit HSMM** | **1.21%** | 50.79% | 35.56% | 88.89% | 0.872 | 13 | 45 | 95.10% |
| **5. FULLY OPTIMIZED PIPELINE** | **69.23%** | **92.31%** | **85.71%** | **100.00%** | **0.944** | **1** | **21** | **97.48%** |

*Ground Truth: 18 Spells.*

---

## 5. Role 4 Real Match Telemetry & Output Formats

The decoding pipeline natively supports real inference probabilities generated by Role 4:
- **Input Parquet**: `role5_probabilities.parquet` containing `['frame_id', 'p_home', 'p_away', 'p_dead']` (199,319 match frames).
- **Match Synchronization**: `sync_offset.json` containing camera synchronization and halves intervals:
  - `HALF_1`: frames 11,185 to 90,651 (~53.0 minutes).
  - `HALF_2`: frames 110,808 to 190,197 (~52.9 minutes).
- **Exported Tactical Spells**:
  - `role5_decoded_spells.parquet`: Full tactical possession spells with microsecond timestamps and phase tagging.
  - `role5_decoded_spells.csv`: Human-readable possession intervals.

### Real Match Execution Results (199,319 Frames)
- **Decoding Speed**: 12.58 seconds via Explicit Duration HSMM Trellis (~15,800 frames/sec).
- **Effective In-Play Time**: 51.3% (68.2 mins).
- **Dead Ball Stoppage Time**: 48.7% (64.7 mins).
- **Home Tactical Possession**: **53.76%** (Half 1: 54.1%, Half 2: 53.5%).
- **Away Tactical Possession**: **46.24%** (Half 1: 45.9%, Half 2: 46.5%).
- **Upstream Noise Smoothing**: 4.59% argmax divergence (chatter eliminated).

---

## 6. Execution Instructions

### A. Production Run on Role 4 Parquet
```bash
python3 main.py
```
Options:
- `--parquet <path>`: Custom input Parquet file (default: `role5_probabilities.parquet`).
- `--sync <path>`: Synchronization JSON file (default: `sync_offset.json`).
- `--out-parquet <path>`: Output spells Parquet file (default: `role5_decoded_spells.parquet`).
- `--out-csv <path>`: Output spells CSV file (default: `role5_decoded_spells.csv`).
- `--viterbi-only`: Run fast dynamic Viterbi decoding (<1s).

### B. Run Synthetic Optimization Benchmark
```bash
python3 main.py --benchmark
```

### C. Run Test Suite
```bash
python3 test_decode.py
```
