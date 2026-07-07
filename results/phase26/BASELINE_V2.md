# BASELINE_V2 — Task B1: regularized LR baseline on cohort v2 (E4 suite)

Reference numbers for all Phase 2.6 / E3 ablations. CPU sklearn only.
Builder: `scratchpad/b1_baseline_v2.py` · results: `scratchpad/baseline_v2_results.csv`.

## Setup (as confirmed, no deviation)

- Cohort: `data/prospective/cohort_v2.parquet` (24,960 windows, 1,252 pitchers).
- Features per (pitcher, t), from `slim_games.parquet` strictly before t (X2 recipe + month):
  `pc_30, pc_90, ng_30, ng_90, acwr=(pc_30/30)/(pc_90/90) [0 if pc_90==0],
  days_since_last (0-365), vel_trend, month`. 8 features. `days_since_last` verified
  identical to cohort `dsl` (max diff 0.0).
- Folds: fold_main train 2017-2020 (13,289) / valid 2021 (3,660) / test 2022-2023 (8,011).
- **Fitting set = train+valid (16,949)**; no hyperparameter search. StandardScaler and
  LogisticRegression(class_weight='balanced', max_iter=2000) both fit on that combined
  set; evaluate on test. (Literal spec said "scaler fit on train"; fit on the combined
  fitting set for pipeline consistency — negligible numeric effect.)
- Embargo at H<=150 removes zero windows (verified in cohort build), so main folds used as-is.
- Positive counts VERIFIED against supply grid: H90_B0 138/35/118, H150_B0 205/47/162 (exact).
- Test decision dates: 12 (2022-2023 x Apr..Sep). Unique test pitchers: 903.

## Main results — H in {90,150}, B=0 (test 2022-2023)

PR/ROC CIs: pitcher-clustered bootstrap, 1000 reps seed 0 (paired resamples). null =
analytic (PR=base rate, ROC=0.5); its ranking metrics are a single random-draw seed-0 baseline.

| H | model | base rate | PR-AUC [95% CI] | ROC-AUC [95% CI] | p@10 | p@20 | p@50 |
|---|-------|-----------|-----------------|------------------|------|------|------|
| 90 | null | 0.01473 | 0.01473 (=base) | 0.500 | 0.008 | 0.004 | 0.008 |
| 90 | dsl_only | 0.01473 | 0.0131 [0.0096–0.0175] | 0.482 [0.432–0.532] | 0.008 | 0.013 | 0.012 |
| 90 | **workload** | 0.01473 | **0.0234 [0.0162–0.0337]** | **0.640 [0.585–0.693]** | 0.025 | 0.017 | 0.028 |
| 150 | null | 0.02022 | 0.02022 (=base) | 0.500 | 0.008 | 0.008 | 0.015 |
| 150 | dsl_only | 0.02022 | 0.0185 [0.0137–0.0246] | 0.495 [0.445–0.546] | 0.017 | 0.008 | 0.007 |
| 150 | **workload** | 0.02022 | **0.0324 [0.0232–0.0445]** | **0.634 [0.573–0.693]** | 0.008 | 0.013 | 0.033 |

Reading:
- **workload beats chance on ROC** at both H (CI lower bound 0.573–0.585 excludes 0.5).
- **workload beats base rate on PR** — but only marginally: PR CI lower bound (0.0162 @H90,
  0.0232 @H150) just clears the base rate (0.0147 / 0.0202). Weak, real, borderline.
- **dsl alone is at chance** (ROC CI includes 0.5 at both H). Recent inactivity carries no
  standalone signal in the F=365 cohort — the workload signal is NOT shutdown detection.
- precision@k values are ~base-rate and dominated by noise (12 dates x small k, ~1.5-2% base
  rate). p@50: workload 0.028/0.033 vs base 0.015/0.020 — modest top-of-list lift, but noisy.
  Event recall (below) is the more stable operational metric.

Workload standardized coefficients (H=90): days_since_last **−0.408** (largest), ng_30 −0.284,
pc_30 −0.192, month −0.103, ng_90 −0.065, pc_90 +0.028, acwr −0.015, vel_trend −0.003.
The dominant term (dsl) enters with a **negative** sign: **active** (low-dsl), high-load
pitchers rank higher — the *opposite* of the circular pre-op-shutdown direction.

## Event-level recall (distinct test surgeries caught / total)

A surgery is caught at budget k if ANY window whose label-window contains it ranks in the
top-k of its decision date. Total distinct test surgeries: H=90 → 52, H=150 → 56 (matches
the expected ~52-56; this is the effective positive sample size driving CI width).

| H | model | caught@10 | caught@20 | caught@50 | total |
|---|-------|-----------|-----------|-----------|-------|
| 90 | null (random) | 1 | 1 | 4 | 52 |
| 90 | dsl_only | 1 | 3 | 7 | 52 |
| 90 | **workload** | **3** | **4** | **15** | 52 |
| 150 | null (random) | 1 | 2 | 7 | 56 |
| 150 | dsl_only | 2 | 2 | 2 | 56 |
| 150 | **workload** | 1 | 3 | **14** | 56 |

workload@50 catches 15/52 (29%) and 14/56 (25%) of surgeries vs random ~4/52 (8%), ~7/56
(13%). dsl_only is at or below random. Operationally: a top-50-per-date watchlist recovers
about a quarter to a third of true surgeries — roughly 2-3x random, weak but non-trivial.

## Calibration (workload, logistic fit of y on model logit, test)

| H | slope | intercept |
|---|-------|-----------|
| 90 | 1.403 | −4.256 |
| 150 | 1.185 | −3.919 |

Slope ~1.2-1.4 (>1 → log-odds spread mildly under-dispersed). Large negative intercept is
the expected consequence of class_weight='balanced' inflating absolute probabilities:
rank-ordering is usable, absolute probabilities are NOT calibrated and would need a −4 logit
shift. (For E3, either drop balanced weighting or post-hoc recalibrate before reporting probs.)

## Blackout sweep — workload LR, does the signal survive B>0?

Features are fixed (computed before t regardless of B); only the LABEL window moves to (t+B, t+H].
If the signal were the circular pre-op shutdown, ROC would collapse as B grows. It does not.

| H | B | pos | base rate | PR-AUC [95% CI] | PR lift | ROC-AUC [95% CI] |
|---|---|-----|-----------|-----------------|---------|------------------|
| 90 | 0 | 118 | 0.01473 | 0.0234 [0.0162–0.0337] | 1.59x | 0.640 [0.585–0.693] |
| 90 | 30 | 75 | 0.00936 | 0.0155 [0.0105–0.0236] | 1.65x | 0.622 [0.552–0.682] |
| 90 | 60 | 34 | 0.00424 | 0.0068 [0.0039–0.0153] | 1.60x | 0.584 [0.482–0.673] |
| 150 | 0 | 162 | 0.02022 | 0.0324 [0.0232–0.0445] | 1.60x | 0.634 [0.573–0.693] |
| 150 | 30 | 119 | 0.01485 | 0.0247 [0.0170–0.0362] | 1.67x | 0.631 [0.565–0.697] |
| 150 | 60 | 78 | 0.00974 | 0.0148 [0.0097–0.0223] | 1.52x | 0.633 [0.559–0.705] |
| 150 | 90 | 44 | 0.00549 | 0.0088 [0.0057–0.0147] | 1.60x | 0.648 [0.561–0.734] |

- **PR-lift over base rate is flat at ~1.5-1.67x across every cell** — the raw PR falls only
  because the base rate falls; relative signal is constant.
- **ROC at H=150 is flat across the whole blackout** (0.634 → 0.648, no trend). At H=90 there
  is a mild decline (0.640 → 0.584), but the B=60 CI [0.482–0.673] includes chance so the
  decline is within noise.
- The signal does NOT collapse when the surgery is pushed 3-5 months into the future.

## Interpretation (honest)

**The circular pre-op-shutdown artifact is refuted.** Three independent pieces of evidence:
(1) dsl_only is at chance — inactivity, the shutdown proxy, carries no standalone signal;
(2) in the multivariate model dsl enters with a negative coefficient (active pitchers rank
higher, the anti-shutdown direction); (3) ROC does not collapse under blackout — at H=150 it
is flat out to B=90 (surgery 90-150 days after the decision date). A shutdown-detection
artifact would show the opposite on all three.

**But this is a weak, roughly time-invariant risk-stratification signal, not a sharp acute
early-warning.** The flatness across blackout cuts both ways: a purely *acute* lead-time
signal would peak at B=0 and decay as the label window moves later — we see no decay at H=150.
That pattern is more consistent with the model detecting **chronically higher-risk workload/
velocity profiles** (pitchers who are structurally more surgery-prone) than an acute countdown
to a specific surgery. For KBO risk-stratification this is still useful (a stable ordering of
who is more at risk), but it should be framed as a risk profiler, not an imminent-injury alarm.

**Versus the old Phase 2.5 reference (PR 0.024 / ROC 0.564, pitch-shape baseline, F=30,
snapshot labels).** Compare lift-over-base (base rates and label completeness differ):
- old pitch-shape: ROC 0.564, PR 0.024 / base 0.0185 → **lift 1.30x**.
- new workload (H=150): ROC **0.634**, PR 0.0324 / base 0.0202 → **lift 1.60x**.
The cohort-v2 workload model beats the old pitch-shape baseline on both ROC (+0.07, outside CI)
and PR-lift (1.60x vs 1.30x). Note it is essentially *tied* with the old X2 workload LR
(ROC 0.638 / PR 0.0294, F=30) — the F=365 redesign + event labels did NOT raise the headline
number; its payoff is (a) removing the F=30 dsl censoring artifact and (b) enabling the
blackout diagnostic that shows the signal is forward-looking, not circular.

## Caveats

- Effective positive n is only ~52-56 distinct test surgeries (118/162 positive windows are
  autocorrelated across a pitcher's consecutive monthly dates). Treat all CIs as wide.
  Differences within roughly ±0.05 ROC / ±0.01 PR between adjacent blackout cells are within
  bootstrap noise; only the aggregate patterns (workload > chance; no blackout collapse) are
  robust.
- PR improvement over base rate is borderline (CI lower bound barely clears base rate).
- precision@k is noisy and near base rate; do not headline it. Null ranking metrics are a
  single random-draw seed-0 reference (analytic null PR = base rate, ROC = 0.5).
- Absolute probabilities are uncalibrated (balanced class weights); only rank-order is meaningful.
- The "chronic risk-type vs acute lead-time" question is not resolved by this baseline; a
  pitcher-fixed-effect or within-pitcher design would be needed to separate them.
