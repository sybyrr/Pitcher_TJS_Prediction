# Data-driven approaches for predicting Tommy John Surgery risk in Major League Baseball pitchers

**Authors:** Bosuk Kang, Minsu Park (corresponding), Angel P. del Pobil, Eunil Park (corresponding)
**Journal:** *Journal of Big Data* (2025) 12:87
**DOI:** 10.1186/s40537-025-01138-1
**Received:** 15 Dec 2024 · **Accepted:** 25 Mar 2025 · **Published online:** 11 Apr 2025
**Code & data:** https://github.com/dxlabskku/TJS_Prediction

> This is a layout-aware Markdown transcription of the paper prepared as a working reference for reproduction. Section order, tables, figure captions, and reported numbers follow the original. Figures themselves are described, not reproduced. Numbers are transcribed from the source; the one internal discrepancy (regression R² 0.79 in the paper body vs. 0.78 in the repo README) is flagged inline.

---

## Abstract

Injury management is critical in all sports. Among injuries, Tommy John Surgery (TJS) is a notable risk for MLB pitchers. Traditional TJS prediction required sensors or video motion capture, impractical during games and limited to predictions very close to injury (e.g., within 30 pitches). This study proposes a deep learning (DL) framework combining **classification** and **regression** tasks. Using MLB pitching data (2016–2023), the classification model detects injury risk **up to 100 days in advance** with an **F1-score of 0.73**, while the regression model estimates time remaining until the player's last pre-surgery game with **R² of 0.79**. An explainable AI (XAI) technique identifies impacting mechanical features, such as a lowered four-seam fastball release point, that accelerate UCL deterioration.

**Keywords:** Injury prediction, Tommy John Surgery (TJS), Big data, Baseball, Deep learning, Classification, Regression

---

## 1. Introduction

- Injuries carry large competitive/financial cost (e.g., EPL: ~£45M/season; every 136 player-absence days ≈ one league-standings point).
- Baseball is more demanding than soccer schedule-wise: MLB ~162 games/season, ~5–7 games/week vs. EPL 38 matches (~1/week). Condensed schedule → cumulative fatigue/injury risk.
- Elbow injuries dominate for pitchers: 59.1% of starters and 58.3% of relievers report elbow injuries. If an elbow injury progresses past rehab it can become UCL damage requiring TJS.
- TJS: ~20.5 month average recovery; ~$1.9M average financial loss per pitcher. TJS incidence has risen steadily since 1974 (Fig. 1).

**Contributions:**
1. First (to authors' knowledge) to specifically predict TJS-leading injuries in MLB pitchers using widely available in-game data (not general elbow injuries or limited-access medical data).
2. Enables prediction up to 100 days in advance; XAI identifies critical mechanical changes.
3. XAI reveals a lowered four-seam fastball release point signals impending injury.

---

## 2. Related work

### 2.1 Traditional elbow-injury studies
- Video analysis of 23 pro pitchers over 3 seasons → injury linked to elevated elbow valgus torque and high shoulder external rotation torque.
- 3D motion of 69 adult pitchers → six key variables incl. elbow flexion influence valgus torque; elbow flexion at release correlates with valgus torque variation.
- Medical imaging: ultrasound of 70 pitchers → ulnohumeral gap > 5.6 mm ≈ sixfold higher UCL-rupture likelihood (AUC 0.77, p = 0.02). Comparative ultrasound of 203 pitchers → TJS group had larger resting joint space (0.5 vs 0.2 mm, p = 0.006) and more hypoechoic foci (57.9% vs 30.4%, p = 0.03).
- Limitation: non-game data; can't capture real-time in-game mechanics.

### 2.2 ML/DL studies
- Ensemble (RF + LogReg + XGBoost) on 4,657 players to predict next-season DL placement: F1 0.70 (hitters) / 0.55 (pitchers); but elbow-specific F1 collapsed to 0.07 / 0.17.
- I3D ConvNet on broadcast video of 20 injured 2017 pitchers → UCL-tear F1 0.74 (LHP) / 0.69 (RHP); but predicts within ~20–30 pitches of injury (too close to be actionable).
- Gap this paper targets: predict earlier (100 days), interpret mechanics.

---

## 3. Dataset

### 3.1 Data source
- MLB pitching data **2016–2023** via **Pybaseball** (Trackman-era metrics). Result: **5,537,981 pitches**, **94 attributes**, **regular-season games only**.
- Players split into TJS vs non-TJS. TJS dates from a public database (Roegele Tommy John Surgery list, tracks MLB + MiLB since 2012). For TJS group, used the **last two seasons before surgery** (min. 2 game appearances). Non-injury players: participated in ≥ 3 of most recent 4 seasons.
- **Total: 620 samples — 101 injured, 519 non-injured.**

**Table 1 — Match frequency EPL vs MLB**

| League | Duration (months) | Total matches | Weekly games |
|---|---|---|---|
| EPL | 9.5 | 38 | ~1 |
| MLB | 7.5 | 162 | ~6 |

**Table 2 — Normal vs. injured cases by number of seasons represented**

| Category | Normal (4 / 3 / 2 seasons) | Injured (4 / 3 / 2 seasons) |
|---|---|---|
| Cases | 518 / 1 / – | 44 / 53 / 4 |
| **Total** | **519** | **101** |

**Table 3 — Metrics by handedness and injury status** (mean ± SD)

| Category | Normal LHP | Normal RHP | Injured LHP | Injured RHP |
|---|---|---|---|---|
| Cases (n) | 149 | 370 | 25 | 76 |
| Height (cm) | 189.5 ± 4.6 | 189.2 ± 5.4 | 188.9 ± 5.0 | 188.5 ± 5.6 |
| Weight (kg) | 97.7 ± 8.9 | 98.5 ± 9.3 | 98.1 ± 9.7 | 97.6 ± 8.7 |
| Fastball speed (mph) | 92.6 ± 2.7 | 93.9 ± 2.5 | 92.8 ± 2.6 | 94.9 ± 2.2 |
| Spin rate (rpm) | 2222 ± 171 | 2289 ± 159 | 2222 ± 131 | 2315 ± 173 |

Fatigue-type variables (total pitch counts, player age) were **excluded** — large individual differences, inconsistently tracked across official/unofficial games, and TJS occurs even under-18.

### 3.2 Pitch metrics and pitch types

**Table 4 — 17 pitching metrics (defined by Baseball Savant)**

| Metric | Description |
|---|---|
| Ax, Ay, Az | Acceleration of the pitch in X, Y, Z |
| Effective_Speed | Derived speed from pitcher's release (hitter's perspective) |
| Pfx_X, Pfx_Z | Movement in X and Z (catcher's perspective) |
| Plate_X, Plate_Z | Ball position in X and Z when crossing home plate |
| Release_Extension | Release extension in feet (Statcast) |
| Release_Pos_X, Release_Pos_Z | Release position in X and Z (catcher's perspective) |
| Release_Speed | Pitch velocity at release |
| Release_Spin_Rate | Spin rate |
| Spin_Axis | Spin axis in 2D X-Z plane, degrees 0–360 |
| Vx, Vy, Vz | Velocity of the pitch in X, Y, Z |

**Table 5 — 6 pitch types** (<10% missing kept): CH (change-up), CU (curve), FC (cutter), FF (four-seam fastball), SI (sinker / two-seam), SL (slider).

→ **17 metrics × 6 pitch types = 102 features per pitcher.**

### 3.3 Preprocessing
1. Pivot Pybaseball records by pitch type → pitch-type-specific columns.
2. Standardize all metrics to right-handed reference (flip LHP sign for x-axis / spin axis).
3. Missing values: entire-column-missing pitcher → global mean; partial → linear interpolation.
4. **Differencing** vs. each player's first-season average (not absolute values).
5. Outliers: drop points beyond ±4.7σ from the metric mean.
6. **Time-series reordering (Fig. 2):** last game (for injured, last game before surgery) = **day 0**; prior records in reverse chronological order → daily-interval time series (days remaining until last game).
7. **Classification:** aggregate into **5-day intervals** (averaged). **Regression:** keep all time points but still 5-day interval for input.

---

## 4. Method

Models: LSTM, CNN-LSTM, Transformer-Encoder, ResNet, ViT (classification); 1D-CNN (regression).

- **Classification window:** analyze data up to **100 days prior to injury**. 100-day choice motivated by nonoperative partial-UCL-tear rehab (~3 months / 12 weeks) — enough lead time for intervention.
- **Regression window:** up to **550 days** prior to the injury event (longer history improved stability).
- Vision models (ResNet, ViT): transform time-series into a **single-channel image** (time on one axis, pitching metrics on the other) — a 2D "feature-time space." Not real images/video.

**Table 6 — Model architectures**

| Model | Configuration |
|---|---|
| ViT | vit_base_patch16_224, input 224, pretrained on ImageNet |
| ResNet | ResNet-50 pretrained on ImageNet |
| Transformer Encoder | 512-dim embeddings, 16 heads, 12 layers |
| CNN+LSTM | Two 1D-CNN layers → 4-layer LSTM (512-dim) |
| LSTM | 4-layer LSTM, 512-dim (paper text also mentions a six-layer bidirectional variant) |

- **Classification loss:** Weighted BCE. In ViT trials, Weighted BCE improved injured-class F1 by ≥ 0.03 vs. standard BCE and SMOTE. Positive-class weight ≈ **5** (≈ 5:1 imbalance).
- **Regression model:** single-channel **1D-CNN** — 4 conv layers (each: BatchNorm → ReLU → MaxPool) → flatten → 4 fully-connected layers → single scalar (days until injury). Sequential models (LSTM/CNN-LSTM/Transformer) all gave R² < 0.1; vision models unsuitable for many-to-many regression.

---

## 5. Experiment

- 10 different random seeds per model.
- Hardware (paper): NVIDIA RTX 4090 (22.5 GB), Intel Xeon @ 2.20 GHz, Python 3.11.11.
  *(Repo README lists NVIDIA L4 22.5 GB, CUDA 12.5, PyTorch 2.5.1 — same 22.5 GB memory class.)*

### 5.1 Classification
- Split 6:2:2 (train/valid/test), stratified.
- Class imbalance: Weighted BCE chosen over SMOTE; Pos Weight = 5. MinMaxScaler; Adam + Cosine Annealing. Grid search over LR, batch size, regularization (Table 7).
- Metrics: Precision, Recall, F1, ROC-AUC (imbalance ≈ 5:1).

**Table 7 — Classification hyperparameters (bold = used)**

| Model | Batch size | Learning rate | L1 Reg. | L2 Reg. | Pos weight |
|---|---|---|---|---|---|
| ViT | **16**, 32, 64, 128 | 1e-6–5e-6 → **4e-6** | 0–3e-6 → **0** | 0–3e-5 → **3e-5** | — |
| ResNet | **16**, 32, 64 | 1e-5–8e-5 → **6e-5** | 0–3e-6 → **1e-6** | 0–1e-5 → **0** | 1, **5** |
| Transformer | 32, **64**, 128 | 1e-6–5e-6 → **3e-6** | 0–3e-7 → **1e-7** | 0–1e-6 → **1e-7** | — |
| CNN+LSTM | 16, **32**, 64 | 9e-7–5e-6 → **2e-5** | 0 | 0 | — |
| LSTM | 32, 64, **128** | 6e-6–1e-5 → **8e-5** | 0 | **1e-5** | — |

### 5.2 Regression
- Injured players only, split 6:2:2, all recorded dates aggregated (players not distinguished). Loss: MSE.
- Metrics: R²; and **100-Day RMSE** (RMSE restricted to players flagged injured-within-100-days by classification stage).

**Table 8 — Regression hyperparameters (bold = used)**

| Model | Batch size | Learning rate | L1 | L2 | Dropout |
|---|---|---|---|---|---|
| 1D-CNN | **16**, 32, 64, 128 | 1e-4–3e-3 → **1e-3** | 0–1e-4 → **0** | 0–9e-6 → **0** | **0**, 0.1, 0.2 |

---

## 6. Results

### 6.1 Classification (Table 9 — averaged over 10 experiments; test = 104 Normal + 20 Injured)

| Model | Class | Precision | Recall | F1 | ROC-AUC | Acc. | Train (s) | Infer (s) | GPU (GB) |
|---|---|---|---|---|---|---|---|---|---|
| **ViT*** | Normal | 0.94 | 0.96 | 0.95 | **0.93** | **90.8** | 217 | 0.64 | 5.18 |
| | Injured | 0.79 | 0.68 | **0.73** | | | | | |
| ResNet* | Normal | 0.93 | 0.95 | 0.94 | 0.88 | 89.3 | 61 | 0.09 | 1.29 |
| | Injured | 0.71 | 0.60 | 0.64 | | | | | |
| Transformer | Normal | 0.91 | 0.82 | 0.86 | 0.78 | 78.3 | 355 | 0.36 | 6.77 |
| | Injured | 0.39 | 0.59 | 0.46 | | | | | |
| CNN+LSTM | Normal | 0.91 | 0.78 | 0.84 | 0.75 | 74.9 | 45 | 0.02 | 2.79 |
| | Injured | 0.35 | 0.61 | 0.44 | | | | | |
| LSTM | Normal | 0.90 | 0.64 | 0.75 | 0.67 | 63.9 | 135 | 0.07 | 5.74 |
| | Injured | 0.24 | 0.62 | 0.35 | | | | | |

`*` = pretrained. Ranking: **ViT > ResNet > Transformer > CNN-LSTM > LSTM.** ViT training ~3× and inference ~7× ResNet, ~4× GPU; total inference for 124 players ≈ 0.64 s (feasible per-game).

### 6.2 Regression
- **R² ≈ 0.79** (paper body; repo README says 0.78), **100-Day RMSE ≈ 95.7**, over 10 runs. Inverse relation between 100-Day RMSE and R².

### 6.3 SHAP interpretation
- Gradient SHAP chosen over LIME (global + local; dependence plots).
- 10 runs → top-10 SHAP features per run → **20 distinct features** consistently important. Retraining on only these 20: R² drops just 0.06, 100-Day RMSE drops 2.2 → 20 features preserve most predictive signal.

**Table 10 — Feature top-10 / top-5 / top-3 SHAP counts (over 10 experiments)**

| Feature | Top10 | Top5 | Top3 |
|---|---|---|---|
| Release_Speed_FF | 10 | 10 | 8 |
| Spin_Axis_FF | 9 | 10 | 10 |
| Release_Extension_SL | 9 | 9 | 6 |
| Release_Pos_X_SI | 9 | 8 | 2 |
| Spin_Axis_SI | 9 | 2 | 0 |
| Release_Spin_Rate_FF | 8 | 5 | 2 |
| Release_Speed_SL | 7 | 3 | 1 |
| Release_Extension_FF | 4 | 1 | 1 |
| Release_Pos_X_SL | 4 | 0 | 0 |
| Release_Pos_X_FF | 4 | 0 | 0 |
| Pfx_Z_CH | 3 | 1 | 0 |
| Release_Pos_X_CH | 3 | 0 | 0 |
| Spin_Axis_SL | 3 | 0 | 0 |
| Release_Pos_Z_CH | 2 | 0 | 0 |
| Release_Extension_CU | 2 | 0 | 0 |
| Release_Pos_Z_CU | 1 | 1 | 0 |
| Release_Pos_Z_SI | 1 | 0 | 0 |
| Release_Spin_Rate_SL | 1 | 0 | 0 |
| Vy0_SL | 1 | 0 | 0 |
| Release_Speed_CH | 1 | 0 | 0 |

Top-3 by average SHAP value — all **four-seam-fastball-associated**: Release_Speed_FF (17.05), Spin_Axis_FF (14.54), Release_Extension_SL (13.41; note SL, but strongly correlated R²=0.59 with Release_Extension_FF).

**Mechanistic reading (as injury nears):**
- **Release_Speed_FF ↓** (r = 0.18, p = 0.07 — small/non-significant but practically relevant decline in final pre-injury season).
- **Spin_Axis_FF → more horizontal** (+~6° by final game; r = −0.61, p < 0.05).
- **Release_Extension_SL ↑** (~0.2 ft; r = 0.70, p < 0.05).
- **Release_Pos_Z_FF ↓** (r = 0.39, p < 0.05) → lowered forearm at release → lower elbow flexion angle → higher valgus torque (supported by cited torque studies: injured 91.6 N·m vs 74.7 N·m, p = 0.013; and valgus-vs-flexion r = −0.36, p = 0.04).

---

## 7. Discussion & Conclusions

- ViT best (F1 0.73, ROC-AUC 0.93), 100-day-ahead classification on game-derived data; 1D-CNN regression R² 0.79.
- SHAP-linked mechanics (vertical release drop ~0.2 ft, spin axis ~6° more horizontal) tie to elevated UCL valgus stress.

**Limitations / future work:**
- Only 620 cases → variance-sensitive metrics (F1); MLB-only → limited generalizability.
- Future: add MiLB / international leagues; multimodal (wearables, biomechanics, video); causal-inference XAI.
- Ethics: false positives could pull a healthy pitcher; expert judgment required.
- Findings are **supportive, not definitive**; Spin_Axis_FF may complement (not replace) MRI.

**Data & code:** https://github.com/dxlabskku/TJS_Prediction

---

## Selected references (as cited above)
- Kang B., Park M., del Pobil A.P., Park E. *Data-driven approaches for predicting Tommy John Surgery risk in MLB pitchers.* J Big Data 12:87 (2025).
- Roegele J. *Tommy John Surgery List.* Public Google Sheet (maintained since 2012).
- Li Z., Li S., Yan X. *Time series as images: ViT for irregularly sampled time series.* NeurIPS 36 (2024).
- Lundberg & Lee. *A unified approach to interpreting model predictions (SHAP).* NeurIPS 30 (2017).
- Dosovitskiy et al. *An image is worth 16×16 words (ViT).* ICLR (2021).
