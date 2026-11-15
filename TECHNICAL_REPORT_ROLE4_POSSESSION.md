# Technical Documentation — Role 4: Hybrid Probabilistic Possession Classification

## 1. The Approach (Model & Decision Architecture)

Role 4 estimates which team controls the ball while play is active. It is designed as a **hybrid probabilistic system**, rather than a single classifier or a collection of hand-written rules. The current architecture combines three complementary components:

1. **Supervised machine learning:** a trained `HistGradientBoosting` classifier estimates the conditional probability of Home versus Away possession from exactly 37 tracking and tactical features.
2. **Quality-aware gating:** tracking observability and reliability signals control how much trust is assigned to frame-level evidence. A directly observed ball position is not treated as equally reliable as a long-gap inferred position.
3. **Evidence Fusion V2:** deterministic, configurable, frame-general football rules use additional competition features to refine the conditional Home/Away estimate. These include causal control differences, pitch-control estimates, player-to-ball distances and time-to-intercept evidence. A positive `control_diff_causal` value represents an advantage for the Home team.

Role 4 is deliberately separated from dead-ball detection. Role 3 supplies the dead-ball probability, `P(D)`, while Role 4 estimates the team in possession **conditional on live play**. Once Evidence Fusion V2 has produced the final conditional H/A split, the probabilities are combined as follows:

\[
P(H) = (1-P(D))\,P(H\mid live)
\]

\[
P(A) = (1-P(D))\,P(A\mid live)
\]

\[
P(D) = P(D)_{Role\ 3}
\]

The result is a three-state probability vector for every frame. It is checked for finite values, valid probability bounds and a sum of one within the configured numerical tolerance. Probabilities are preserved for the downstream temporal decoder rather than being converted to hard labels inside Role 4.

This decomposition reflects the competition definition: a graded frame is Home, Away or Dead Ball. A loose ball or a challenge is not an independent possession class. A touch, deflection or aerial duel without subsequent control is not by itself a possession change. A stoppage is `D`, regardless of which team is expected to restart.

---

## 2. Assets Used, Assets Dropped, and Rejected Approaches

### 2.1. Assets Used and How

* **Competition tracking and tactical feature Parquet:** the frame-level competition dataset is the inference input and is expected to contain exactly 111 columns for one match. The 37 features shared with the training schema are passed to HGB. Additional competition-only signals remain available to quality gating and Evidence Fusion V2; they are not silently added to the HGB feature vector.
* **SkillCorner Open Data:** the external supervised training source. The project training package used for this model contains **20 distinct matches**, sampled at 10 Hz. The target has two live-play classes: Home and Away. The model uses 37 features (`model_features_v2`). The upstream public README currently says 10 matches; the count of 20 in this report refers to the actual 20 unique `match_id` values in the project training package/manifest used for training.
* **Role 2 tracking and physics outputs:** provide frame-level context for assessing observation reliability. Where available, observation status and uncertainty information distinguish direct ball observations from short interpolations and longer inferred gaps. These signals describe the quality of the evidence; they are not treated as possession ground truth.
* **Role 3 dead-ball probabilities:** provide `P(D)` so that the H/A classifier does not have to force every frame into one team during a stoppage.
* **Evidence-fusion features:** causal control differences (`control_diff_causal50`, `control_diff_causal100`), pitch-control estimates, nearest-player distances, minimum time-to-intercept and relevant goalkeeper/context features provide additional football-specific evidence.

### 2.2. Dropped Assets & Rejected Technical Approaches (And Why)

The following alternatives were explored and not selected as the final Role 4 solution:

| Rejected approach / asset | Reason for rejection |
| :--- | :--- |
| **1. Pure white-box possession rules** | The first deterministic baseline relied on manually designed thresholds and state rules. It proved insufficient as a complete frame-level classifier. Possession is not always equivalent to the nearest player or the smallest player-to-ball distance: loose balls, deflections, aerial duels, challenges, tracking gaps and brief touches make isolated geometric rules brittle. The rules remain useful as interpretable supporting evidence, but they did not provide enough flexibility to model the combinations of spatial, kinematic and temporal signals required for robust Home/Away classification. |
| **2. Gaussian Mixture Model (GMM)** | The unsupervised mixture model did not produce a stable mapping from mixture components to the semantic Home/Away labels. Patterns associated with one team could be split across several clusters, and cluster identifiers have no intrinsic football meaning. Converting the resulting components into consistent team-possession labels was therefore ambiguous and unsuitable for frame-level output. |
| **3. Logistic Regression** | Logistic Regression was evaluated on the supervised feature representation but produced weaker experimental results than `HistGradientBoosting`. Its linear decision boundary was less expressive for the non-linear relationships and interactions between player proximity, movement, team shape and control-related features. HGB was selected as the main supervised estimator. No unverified numerical performance gain is claimed here. |
| **4. Candidate Metrica Sports open-data source** | An exploratory attempt to use the candidate Metrica Sports open dataset as a modelling source was discontinued because the inspected data was not sufficiently reliable for the intended training setup. It did not provide a strong enough quality basis to replace the selected external training source. This is a project-specific assessment of the inspected files, not a general claim about every Metrica Sports dataset. |

The rejected methods were not used to create manual frame labels for the competition match. The final system remains algorithmic and reproducible.

---

## 3. Training Data, Provenance & Licences

### 3.1. Supervised training source and label definition

The supervised classifier is trained on **SkillCorner Open Data**, not on the competition match. The target is binary and represents the team in possession while the ball is in play:

| Target value | Meaning |
| :--- | :--- |
| `0` | Home team in possession |
| `1` | Away team in possession |

The project training manifest specifies:

* **Data volume:** 20 unique matches (`match_id` values) in the training package used for this project.
* **Sampling rate:** 10 Hz.
* **Model input:** 37 features (`model_features_v2`).
* **Target:** frame-level Home/Away possession (`possession.group` in the upstream tracking schema, mapped to the project's numeric target values).
* **Estimator:** `HistGradientBoosting`.

The count of 20 is based on the actual training package/manifest used by the project. The upstream SkillCorner Open Data README currently describes 10 matches. These two counts do not agree. For this report, the training-data amount is stated as 20 because that is the number of distinct matches in the package used to build the model, not because the upstream README says so. The public repository should preserve the project manifest (including the exact 20 match IDs and the source revision/download reference) so that organisers can identify precisely which data were used.

### 3.2. Data Source, Attribution, Licence and Label Provenance

**Data provider:** SkillCorner. The SkillCorner Open Data repository contains broadcast tracking data collected from match footage using computer vision and machine learning. The tracking records include a `possession` field with `player_id` and `group` attributes, where `group` identifies the Home or Away team in possession.

**Open-data release partner:** PySport. According to the upstream README, the dataset was made publicly available through a joint initiative between SkillCorner and PySport. The README explicitly requests that users credit SkillCorner when using the data.

**Copyright and licence notice:** The upstream `SkillCorner/opendata` repository contains an MIT `LICENSE` file with the following copyright notice:

`Copyright (c) 2020 SkillCorner. https://skillcorner.com`

The licence grants permission to use, copy, modify and distribute the covered "Software" and associated documentation files, subject to retaining the copyright and permission notices. The repository is marked as MIT-licensed on GitHub.

However, the licence text does not identify individual authors, developers, data annotators or label creators. It identifies SkillCorner in the copyright notice. Furthermore, although the repository includes raw tracking data, the wording of the licence does not unambiguously clarify whether every raw dataset file and all underlying broadcast-related rights are covered by the MIT grant. We therefore retain the original licence and attribution without making an unsupported claim that all underlying match and tracking data are independently licensed under MIT.

**Possession-label provenance:** The public documentation defines `possession.group` as the field indicating which player/team (Home or Away) is in possession. It does not identify individual annotators, a named annotation team, or a sufficiently detailed procedure explaining how these possession labels were generated and verified. Although SkillCorner states that its broadcast tracking is produced using computer vision and machine learning, the documentation does not explicitly establish whether the possession field itself is fully algorithmically generated, manually annotated, or produced through a combination of automated processing and human review.

For this reason, this report attributes the dataset to SkillCorner as its documented provider, credits PySport as the open-data release partner, and does not invent individual author or annotator names. The precise provenance of the possession labels remains undocumented in the publicly accessible source material and should be confirmed with SkillCorner if a more detailed attribution is required.

**Dataset volume used in this project:**
- 20 unique matches (`match_id` values) in the actual project training package.
- Sampling frequency: 10 Hz.
- Supervised target: Home/Away possession, mapped from the upstream `possession.group` field.
- Feature representation: 37 features (`model_features_v2`).

The upstream public README currently describes 10 matches. The 20-match figure above refers to the actual training package and manifest used in this project. The project repository should retain the corresponding manifest and source references to make the exact training-data subset inspectable.

**Attribution:**

> Training data: SkillCorner Open Data, provided by SkillCorner and released through a joint initiative with PySport. The project used 20 unique matches at 10 Hz. The upstream repository contains an MIT licence notice with the copyright attribution `Copyright (c) 2020 SkillCorner`. Individual authors or annotators of the `possession.group` labels are not identified in the available public documentation.
>
> Source: https://github.com/SkillCorner/opendata  
> Licence: https://github.com/SkillCorner/opendata/blob/master/LICENSE

### 3.3. Competition-data separation

The competition match is used for inference only. No hand-labelled frames from the competition match are used to train or validate Role 4. The public five-minute ground-truth window is reserved for checking that the data pipeline, frame alignment and output format function correctly; it is not used to fit the classifier or tune its parameters. The private full-match ground truth and any information derived from it are not part of the modelling process.

---

## 4. Quality Gating & Evidence Fusion V2

### 4.1. Quality-aware gating

Tracking quality is not constant across frames. A ball coordinate may be directly observed, interpolated across a short gap, or inferred during a longer occlusion. Treating all such coordinates as equally reliable can make a model or rule overconfident precisely when its underlying evidence is weakest.

Role 4 therefore uses quality and observability information from the upstream tracking pipeline to control the influence of frame-level evidence. Quality gating does **not** create a new possession label. Its purpose is to reduce or increase the trust assigned to evidence according to how well that evidence is supported by tracking and camera observations.

The pipeline separates three responsibilities:

* **HGB:** estimates the conditional Home/Away probability from the 37-feature training representation.
* **Quality gating:** assesses the reliability of the current observations and controls how strongly uncertain frame-level signals should influence the estimate.
* **Evidence Fusion V2:** adds football-specific evidence available in the competition dataset but not included in the 37-feature HGB input.

### 4.2. Deterministic evidence fusion

The evidence layer uses general, frame-independent relationships between tracking signals and team control. The configured evidence includes:

* **Causal control difference:** `control_diff_causal50` and `control_diff_causal100` provide directional evidence about local team control. Positive values support Home; negative values support Away.
* **Pitch control:** relative spatial control estimates provide team-level context beyond the identity of the nearest player.
* **Proximity and interception:** nearest-player distance and minimum time-to-intercept help distinguish immediate ball access from more distant pressure or pursuit.
* **Contextual signals:** goalkeeper and other tactical/geometry features can support or weaken the interpretation of a local control estimate.

The evidence layer is complementary to the trained classifier. It does not replace HGB with one hard threshold and does not introduce labels tied to particular timestamps, frames or observed episodes in the competition match. The weights and thresholds are configuration-driven and should remain committed with the source code, so the full decision logic is inspectable and reproducible. They must not be tuned against private ground truth or selected by manually studying particular competition-match outcomes.

### 4.3. Combining Home/Away possession with dead-ball probability

Role 4 estimates the conditional Home/Away split; Role 3 estimates dead-ball probability. The wrapper applies the three-state fusion:

```text
P(H) = (1 - P(D)) * P(H | live)
P(A) = (1 - P(D)) * P(A | live)
P(D) = P(D) from Role 3
```

The wrapper is deliberately **fail-closed**. Before final fusion, it requires confirmation that (1) Role 3's `P(D)` means `out_of_play_only`, not contested in-play situations, and (2) Role 4's H/A output is interpreted as conditional on live play. These confirmations can be stored in `config.py` or passed as explicit CLI flags. The flags only authorise the agreed semantics; they do not calibrate or retrain the model.

---

## 5. Possession Semantics & Rule Boundaries

The Role 4 probability output represents three states:

* **Home (`H`):** Home possession while the ball is in play.
* **Away (`A`):** Away possession while the ball is in play.
* **Dead Ball (`D`):** the ball is out of play. This is a stoppage state, not possession by the team that last touched the ball or the team expected to restart.

A live-ball contest is not automatically a possession change. An opponent touch, a deflection that does not result in control, or an aerial duel without subsequent control is not by itself sufficient to establish a turnover. The downstream Role 5 temporal decoder is responsible for interpreting the probability sequence and producing the final discrete competition labels, including `X` where appropriate. Role 4 does not create an `X` class.

---

## 6. Limitations

1. **Domain shift between datasets:** the classifier is trained on external SkillCorner tracking and applied to the competition tracking pipeline. Differences in camera geometry, tracking behaviour, sampling, feature distributions and match context can reduce transfer performance.
2. **Training-source documentation:** the project training package contains 20 unique matches while the public SkillCorner README currently says 10. The project manifest should be committed with the exact match IDs and source revision. Public documentation also does not fully specify the annotation workflow or named annotators for `possession.group`; that should be confirmed with SkillCorner.
3. **Licence scope / underlying rights:** the SkillCorner Open Data repository contains an MIT licence notice, but its wording refers to software and associated documentation. The exact terms for a separately exported training package and any underlying broadcast rights should be checked rather than inferred from the repository badge alone.
4. **Unobserved ball states:** when the ball is occluded or inferred, the system must rely more heavily on player geometry and contextual evidence. A dense scramble or simultaneous camera occlusion can still make the true controlling team ambiguous.
5. **Rule-weight sensitivity:** deterministic evidence can improve interpretability and robustness, but poorly chosen thresholds or weights can bias predictions. The configuration must remain general-purpose and must not be selected using private ground truth or tuned against the public five-minute window.
6. **Probability calibration:** better classification performance does not automatically guarantee perfectly calibrated probabilities. The three-state probabilities should be interpreted as model estimates; any calibration change must use permitted external validation data.

---

## 7. Reproduction — One-Command Role 4 Inference

### 7.1. Inputs and dependencies

The `run_role4_pipeline.py` wrapper executes the complete Role 4 inference pipeline. It does not train, refit, recalibrate or modify the HGB model. Training is a separate development step; the trained model artifact must already exist before inference.

The pipeline requires:

- **Competition Parquet:** the full competition tracking dataset, expected to contain 111 columns and represent one match.
- **Role 3 Parquet:** the output of Role 3, containing the dead-ball probability `P(D)` and a frame identifier.
- **Trained HGB artifact:** `role4_hist_gradient_boosting.joblib`, containing the fitted estimator and its model feature schema.
- **Runtime configuration:** `src/possession/config.py`, containing identifier settings, probability-fusion semantics and Evidence Fusion V2 configuration.
- **Python dependencies:** the packages listed in `requirements_role4.txt`.

Create the virtual environment and install the dependencies using `uv`:

```powershell
uv venv
uv pip install --python .venv\Scripts\python.exe -r requirements_role4.txt
```

Run the pipeline from the repository root using the Python interpreter with the installed dependencies. The example below uses `python` from the active project environment.

Required packages include `pandas`, `numpy`, `pyarrow`, `scikit-learn` and `joblib`.

### 7.2. Wrapper execution

The following command executes the complete Role 4 pipeline, from loading the competition data and the existing model to producing the final three-state probability output:

```powershell
python src/possession/inference/run_role4_pipeline.py `
  --competition-parquet data/interim/stage1_tracking.parquet `
  --role3-parquet data/interim/stage2_deadball.parquet `
  --model role4_artifacts/role4_hist_gradient_boosting.joblib `
  --config src/possession/config.py `
  --output role4_artifacts/role5_probabilities.parquet `
  --confirm-role3-out-of-play-only `
  --confirm-role4-conditional-live
```

The two semantic confirmation flags explicitly confirm that:

1. Role 3's `P(D)` represents out-of-play situations only, not contested in-play situations.
2. Role 4's Home/Away probabilities represent possession probabilities conditional on live play.

These confirmations are required before the final three-state fusion is performed. They can alternatively be recorded in `config.py`, in which case the corresponding CLI flags are not necessary.

The wrapper enables three-state fusion internally. No separate `--enable-three-state-fusion` argument is required.

For non-standard input column names, the wrapper also supports explicit overrides, including:

- `--competition-frame-column`
- `--role3-frame-column`
- `--competition-match-column`
- `--role3-match-column`
- `--role3-probability-column`

The Role 3 probability column is taken from `config.py` by default. The output frame column is named `frame_id` by default. If the output file already exists, the wrapper stops unless `--overwrite` is provided.

### 7.3. Model artifact, feature selection and inference

The HGB model is loaded from the `.joblib` artifact using `joblib.load()`. The artifact is a dictionary containing:

- `estimator` — the fitted `HistGradientBoosting` classifier;
- `feature_names` — the exact ordered list of 37 model features used during training;
- `model_name` — the model identifier.

The feature manifest (`skillcorner_feature_manifest.json`) is used during training to define the model feature set. The selected feature names are saved into the `.joblib` artifact together with the fitted estimator.

**Inference does not reload or require the original feature manifest.** All model-specific information required to select the HGB inputs and run the fitted classifier is obtained from the `.joblib` artifact.

The inference code validates that the artifact contains exactly 37 unique feature names and that every required feature is present in the competition Parquet. It then selects the model inputs in the exact order stored in the artifact:

```python
X = df.loc[:, features].copy()
```

Only these 37 columns are passed to `estimator.predict_proba()`. The remaining competition columns are retained in the full DataFrame and can be used independently by Evidence Fusion V2, quality gates and other configured processing.

The runtime configuration is loaded separately from `config.py`. The competition tracking values come from the competition Parquet, while the dead-ball probabilities are read from the Role 3 Parquet and joined to the competition frames.

This separation makes the model artifact self-contained with respect to the trained estimator and its feature schema. The original training manifest is required to reproduce the training process, but not to execute inference using an existing `.joblib` artifact.

### 7.4. Join, inference and probability fusion

The wrapper performs the following steps:

1. Loads `config.py`, the full competition Parquet, the Role 3 Parquet and the trained HGB `.joblib` artifact.
2. Loads the fitted estimator and its stored list of 37 model features from the `.joblib` file.
3. Resolves the competition frame identifier and the corresponding Role 3 frame identifier using configured names, aliases or explicit CLI overrides.
4. Joins the Role 3 probabilities to the competition data using `match_id + frame_id` when both files provide a match identifier. If a shared match identifier is unavailable, a frame-only join is permitted only when both inputs unambiguously represent one match.
5. Validates join-key uniqueness and complete Role 3 coverage. Missing or duplicated frame identifiers and unmatched frames cause the pipeline to fail. The wrapper does not use row order as a substitute for a missing frame identifier.
6. Selects exactly 37 HGB input features using the ordered feature list embedded in the `.joblib` artifact and generates conditional Home/Away probabilities.
7. Applies the configured quality-aware Evidence Fusion V2 rules using additional competition features. These rules refine the conditional H/A probabilities without adding the extra competition columns to the HGB feature vector.
8. Combines the conditional H/A probabilities with Role 3's `P(D)` to obtain the final three-state probability vector:

   ```text
   P(H) = (1 - P(D)) * P(H | live)
   P(A) = (1 - P(D)) * P(A | live)
   P(D) = P(D) from Role 3
   ```

9. Validates that the output frame identifiers are present and unique, all probabilities are finite and within `[0, 1]`, and `P(H) + P(A) + P(D) = 1` within the configured numerical tolerance.
10. Writes the compact probability Parquet and a JSON execution report. Temporary intermediate files are removed automatically.

The wrapper is designed for one match. It expects the full competition schema configured for 111 columns and verifies that the inference result represents a single match. It does not insert a synthetic `match_id` when the input represents one match without that column.

### 7.5. Outputs and Role 5 integration

The default output is a compact Parquet containing exactly four columns and one row per frame:

```text
frame_id
p_home
p_away
p_dead
```

The output file is written to:

`role4_artifacts/role5_probabilities.parquet`

The wrapper also creates an execution report alongside the Parquet:

`role4_artifacts/role5_probabilities_report.json`

The report records the input and output paths, number of processed frames, input and output column counts, resolved frame identifier, Role 3 probability source column, semantic confirmations, model name, number of model features, Evidence Fusion results and maximum probability-sum error.

The wrapper uses a temporary full intermediate Parquet internally to perform inference and final validation. This intermediate file is automatically removed after execution. The default output therefore contains only the compact probability contract, not the original 111 competition columns.

An optional full enriched Parquet can be requested with:

```powershell
--full-enriched-output role4_artifacts/role5_full_context.parquet
```

When this option is enabled, the wrapper additionally saves the full competition DataFrame with the final `p_home`, `p_away` and `p_dead` probabilities. This can be useful when the downstream Role 5 implementation requires the original tracking context and probability outputs in one DataFrame. Otherwise, Role 5 can join the compact probability output back to the original competition data using the frame identifier.

Role 4 does not produce class `X`. The final discrete label decoding, temporal consistency and any abstention handling remain the responsibility of Role 5. The Role 4 wrapper only provides the validated frame-level probability estimates required by the downstream component.

---

## 8. References

* SkillCorner — [Open Data documentation](https://skillcorner.github.io/opendata/): tracking-data structure, possession field, joint release initiative with PySport, attribution request and stated tracking limitations.
* SkillCorner — [Open Data GitHub repository](https://github.com/SkillCorner/opendata): source repository for the public dataset and code.
* SkillCorner — [MIT LICENSE](https://github.com/SkillCorner/opendata/blob/master/LICENSE): upstream copyright and MIT notice (`Copyright (c) 2020 SkillCorner`).