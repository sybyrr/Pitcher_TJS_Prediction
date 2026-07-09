# R1 — Rolling-Origin Robustness Re-evaluation (+ M-role confirmation)

**Purpose.** The 2022-23 main test fold has been inspected repeatedly (selection-bias
risk) and spans era shifts (COVID-shortened 2020 in train, pitch-clock 2023 in test).
This block checks that (i) the headline prospective signal and (ii) the M-role gain are
not artifacts of that single fold, by re-fitting on an expanding origin and testing each
season 2021/2022/2023 separately.

**Protocol (frozen, unchanged).** StandardScaler + LogisticRegression(class_weight=
'balanced', max_iter=2000); pitcher-clustered bootstrap 1000 resamples, seed 0, **shared
across models** for paired CIs; event-level recall@k per decision date via
next_surgery_date grouping. Reference implementations:
`results/phase26/scripts/m0doubleprime.py`, `role_models.py`.

**Design.** For each test year Y in {2021,2022,2023}: fit set = ALL windows with
`t.year < Y` (no separate valid — no hyperparameter search, consistent with the frozen
protocol's train+valid fitting); test = `t.year == Y`. Embargo: fit windows kept only if
`t + H(days) <= Y-04-01`. Models M0dp (5 feats) and M-role (M0dp + start_share, 6 feats),
H in {90,150}, B=0.

Feature recipes are byte-identical to the frozen baselines: M0dp = {pc_chronic (pc90/90),
pc_acute_dev (pc30/30 - pc90/90), days_since_last, vel_trend, month}; M-role adds
start_share = fraction of trailing-365d games (strictly before t) with total_pitches>=50.

---

## Step 0 — main-fold anchor reproduction (pipeline check, tol 3e-4)

Fit train+valid (2017-2021), test 2022-23. All four anchors reproduce within the
documented L2-geometry whisker → pipeline unchanged.

| model  | H   | ROC (repro) | ROC anchor | dROC     | PR (repro) | PR anchor | dPR      | verdict |
|--------|-----|-------------|------------|----------|------------|-----------|----------|---------|
| M0dp   | 90  | 0.615687    | 0.615557   | +1.3e-04 | 0.022712   | 0.022700  | +1.2e-05 | MATCH   |
| M-role | 90  | 0.643680    | 0.643680   | +2.0e-07 | 0.035973   | 0.035970  | +3.0e-06 | MATCH   |
| M0dp   | 150 | 0.618917    | 0.619131   | -2.1e-04 | 0.030076   | 0.030203  | -1.3e-04 | MATCH   |
| M-role | 150 | 0.643775    | 0.643770   | +4.7e-06 | 0.046995   | 0.047000  | -4.9e-06 | MATCH   |

**Embargo:** removes **0** fit windows in every rolling fold at H<=150 (as expected —
the latest fit window is t=Sept-1 of Y-1, so t+150d ≈ late-Jan of Y ≤ Y-04-01).

---

## Fold table — per (H, test-year, model)

`distinct_surg` = unique (pitcher, surgery_date) events among positive test windows
(= evrec denominator). CIs are pitcher-clustered bootstrap 95%.

### H = 90 (B=0)

| Y    | model  | n / pos / distinct | base    | ROC [95% CI]              | PR [95% CI]               | evrec@10/20/50 |
|------|--------|--------------------|---------|---------------------------|---------------------------|----------------|
| 2021 | M0dp   | 3660 / 35 / 17     | 0.00956 | 0.5909 [0.4955, 0.6874]   | 0.0139 [0.0071, 0.0306]   | 0 / 1 / 3      |
| 2021 | M-role | 3660 / 35 / 17     | 0.00956 | 0.6303 [0.5497, 0.7041]   | 0.0143 [0.0076, 0.0264]   | 1 / 2 / 3      |
| 2022 | M0dp   | 4102 / 58 / 24     | 0.01414 | 0.6652 [0.5822, 0.7463]   | 0.0269 [0.0161, 0.0504]   | 1 / 5 / 12     |
| 2022 | M-role | 4102 / 58 / 24     | 0.01414 | 0.6880 [0.5855, 0.7786]   | 0.0358 [0.0177, 0.0736]   | 4 / 5 / 7      |
| 2023 | M0dp   | 3909 / 60 / 28     | 0.01535 | 0.5721 [0.4661, 0.6663]   | 0.0210 [0.0130, 0.0344]   | 4 / 5 / 7      |
| 2023 | M-role | 3909 / 60 / 28     | 0.01535 | 0.6025 [0.5030, 0.7008]   | 0.0400 [0.0163, 0.1122]   | 2 / 2 / 7      |

### H = 150 (B=0)

| Y    | model  | n / pos / distinct | base    | ROC [95% CI]              | PR [95% CI]               | evrec@10/20/50 |
|------|--------|--------------------|---------|---------------------------|---------------------------|----------------|
| 2021 | M0dp   | 3660 / 47 / 17     | 0.01284 | 0.6103 [0.5219, 0.7120]   | 0.0182 [0.0101, 0.0351]   | 0 / 0 / 1      |
| 2021 | M-role | 3660 / 47 / 17     | 0.01284 | 0.6384 [0.5490, 0.7471]   | 0.0210 [0.0105, 0.0402]   | 1 / 1 / 3      |
| 2022 | M0dp   | 4102 / 79 / 27     | 0.01926 | 0.6396 [0.5728, 0.7140]   | 0.0294 [0.0180, 0.0458]   | 0 / 0 / 7      |
| 2022 | M-role | 4102 / 79 / 27     | 0.01926 | 0.6728 [0.5779, 0.7672]   | 0.0395 [0.0216, 0.0732]   | 4 / 4 / 7      |
| 2023 | M0dp   | 3909 / 83 / 29     | 0.02123 | 0.5869 [0.4818, 0.6850]   | 0.0301 [0.0188, 0.0474]   | 3 / 6 / 7      |
| 2023 | M-role | 3909 / 83 / 29     | 0.02123 | 0.6009 [0.5000, 0.7083]   | 0.0440 [0.0246, 0.0847]   | 2 / 7 / 8      |

---

## Paired deltas — M-role minus M0dp (shared pitcher resamples, nboot=1000)

Point = full-sample difference; CI = pitcher-clustered bootstrap 95%.

| Y    | H   | dROC point | dROC 95% CI            | dPR point  | dPR 95% CI              | sign      |
|------|-----|------------|------------------------|------------|-------------------------|-----------|
| 2021 | 90  | +0.0394    | [-0.0601, +0.1445]     | +0.000407  | [-0.01281, +0.00812]    | incl 0    |
| 2022 | 90  | +0.0229    | [-0.0638, +0.1125]     | +0.008883  | [-0.01310, +0.04011]    | incl 0    |
| 2023 | 90  | +0.0304    | [-0.0243, +0.0912]     | +0.019065  | [-0.00037, +0.08061]    | incl 0*   |
| 2021 | 150 | +0.0282    | [-0.0570, +0.1066]     | +0.002727  | [-0.00906, +0.01557]    | incl 0    |
| 2022 | 150 | +0.0332    | [-0.0342, +0.1027]     | +0.010143  | [-0.00426, +0.03827]    | incl 0    |
| 2023 | 150 | +0.0140    | [-0.0355, +0.0629]     | +0.013968  | [-0.00016, +0.04427]    | incl 0*   |

`*` = CI lower bound essentially at 0 (−3.7e-04 and −1.6e-04); nominally includes 0 but
only marginally. **All 6 folds: dROC point > 0 and dPR point > 0 (unanimous direction).**
No fold's paired CI excludes 0 in either direction (M-role is never individually
significant per single season, but never significantly worse either).

---

## Era notes

- **COVID-shortened 2020 in the fit set (affects Y=2021, and all later folds).** Y=2021
  (fit 2017-2020) is the weakest M0dp fold at H=90 (ROC 0.591) and event recall collapses
  (M0dp evrec@50 = 3/17 at H=90, 1/17 at H=150). The COVID season in train does not break
  the signal — ROC stays > 0.55 — but 2021 is the lowest-power year (only 17 distinct
  surgeries, base rate ~0.96%). Signal survives but is noisy here.
- **Pitch-clock 2023 as the test year.** M0dp ROC drops to its minimum across all folds
  (0.572 at H=90, 0.587 at H=150) — consistent with a mild era-shift degradation of the
  pure-workload model, but still above 0.55. Notably the **M-role gain is LARGEST in 2023**
  (dPR +0.019 / +0.014; M-role PR jumps to 0.040 / 0.044), i.e. the role feature partially
  compensates for the workload model's 2023 slippage. This is the opposite of an artifact:
  the role signal is most useful precisely in the out-of-distribution year.
- **2022 is the strongest fold** (M0dp ROC 0.665 at H=90) — the year that dominates the
  combined 2022-23 main anchor. The main-fold ROC ~0.62-0.64 is therefore a blend of a
  strong 2022 and a weaker 2023, not a single lucky slice.

---

## Pre-specified criteria verdict

**(a) M0dp and M-role ROC > 0.55 in ALL folds → PASS.** Minimum across all 12 (model ×
year × H) cells is M0dp Y=2023 H=90 at ROC 0.5721; every cell exceeds 0.55.

**(b) M-role dPR point > 0 in >=2/3 folds AND no fold with dPR CI excluding 0 downward
→ PASS (M-role confirmed).** dPR point > 0 in **3/3 folds at each H (6/6 overall)**; zero
folds have a dPR CI excluding 0 downward. Holds under either reading ("2/3 folds"
per-H or pooled across all 6). The gain is directionally unanimous and monotone-ish in fit
size, though no single season reaches per-fold significance.

**Overall verdict: anchor=PASS, (a)=PASS, (b)=PASS.** The headline prospective signal and
the M-role increment are both robust to rolling-origin re-evaluation; neither is an
artifact of the 2022-23 selection or of the era boundaries.

---

## Caveats

1. **Low event counts per fold** (17-29 distinct surgeries) → wide bootstrap CIs; per-fold
   ROC CIs routinely span ~0.5, so single-year ROC point estimates are imprecise. The
   robustness claim rests on the *consistency of direction across folds*, not on any one
   fold being individually significant.
2. **M-role never individually significant per season** (all paired CIs include 0). The
   confirmation is a directional/point-estimate criterion (pre-specified), not a per-fold
   significance claim. The two 2023 dPR CIs graze 0 from below (−1.6e-4, −3.7e-4).
3. **No embargo bite at H<=150** — the check is a formality here (0 windows removed); it
   would only matter at longer horizons (H=365) or finer within-season decision spacing.
4. **Expanding-origin fit sets differ from the main anchor.** Y=2022 fits the same
   2017-2021 set as the main fold but tests 2022 alone (not 2022+2023); Y=2023 additionally
   folds 2022 into the fit set. Numbers therefore legitimately differ from the anchors;
   only Step-0 is a pipeline-identity check.
5. **start_share is a KBO-public-tier feature** (games with >=50 pitches), so the confirmed
   M-role gain does not itself argue for tracking-data access — consistent with the
   prior finding that survival signal lives in the open tier.
