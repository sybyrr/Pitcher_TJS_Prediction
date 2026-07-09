# R2 — Age covariate on top of M-role

Script `r2_age.py` (frozen protocol adapted **verbatim** from
`results/phase26/scripts/role_models.py`: StandardScaler + LogisticRegression
(class_weight='balanced', max_iter=2000), fit on fold_main train+valid, test 2022–23;
1000 pitcher-clustered bootstrap resamples, seed 0, **shared across models** for paired
CIs; event recall via `next_surgery_date` grouping). Results: `r2_results.csv`.

**Purpose.** Age is a public, legitimate TJS risk factor absent from the model. Two
questions: (1) does adding age improve held-out prediction over M-role at ~52–56 test
events? (2) does the chronic-usage story partially *proxy* age (i.e. does age absorb the
workload coefficients)?

## Birthdate coverage
MLB StatsAPI `/api/v1/people?personIds=…&fields=people,id,birthDate`, 13 batches of ≤100
ids, ~1 s apart. **Coverage = 1252/1252 (100.00%)** — every cohort pitcher resolved on
the first pass. **No fallback, no imputation, no missing indicator** (chadwick fallback
and fit-set-median imputation coded but never triggered). Cache: `birthdates.csv`.
`age = (t − birthDate).days / 365.25`, computed per (pitcher, t). Age range 19.4–46.3,
median 29.4, mean 29.9; 0 absurd/negative values, 0 NaN after merge.

## Anchor reproduction (must hit before new numbers; tol ~3e-4)
| model | H | ROC (repro / anchor / Δ) | PR (repro / anchor / Δ) |
|---|---|---|---|
| M0dp   | 90  | 0.615687 / 0.615557 / **+1.30e-4** | 0.022712 / 0.022700 / +1.17e-5 |
| M0dp   | 150 | 0.618917 / 0.619131 / **−2.14e-4** | 0.030076 / 0.030203 / −1.27e-4 |
| M-role | 90  | 0.643680 / 0.643680 / **+2.0e-7** | 0.035973 / 0.035970 / +3.0e-6 |
| M-role | 150 | 0.643775 / 0.643770 / **+4.7e-6** | 0.046995 / 0.047000 / −4.9e-6 |

M-role reproduces the task anchor **essentially exactly** (Δ < 5e-6 — same pc_30/pc_90
basis, no reparametrization). M0dp matches within the documented L2-geometry whisker
(< 3e-4). Pipeline verified sane (days_since_last == cohort.dsl, max diff 0.0).

## Results (test set; 95% pitcher-clustered bootstrap, 1000/1000 valid)
### H=90 (52 distinct test surgeries)
| model | PR-AUC [95% CI] | ROC-AUC [95% CI] | evrec @10/20/50 |
|---|---|---|---|
| M-role     | 0.03597 [0.01989, 0.07080] | 0.64368 [0.57643, 0.70933] | 6 / 7 / 13 |
| M-role+age | 0.03397 [0.01920, 0.06679] | 0.63944 [0.56577, 0.71671] | 4 / 8 / 15 |

### H=150 (56 distinct test surgeries)
| model | PR-AUC [95% CI] | ROC-AUC [95% CI] | evrec @10/20/50 |
|---|---|---|---|
| M-role     | 0.04700 [0.02960, 0.07862] | 0.64377 [0.57859, 0.70722] | 8 / 11 / 15 |
| M-role+age | 0.04995 [0.03010, 0.08539] | 0.66683 [0.59891, 0.73782] | 6 / 9 / 18 |

## Paired deltas — (M-role+age) minus (M-role), same 1000 resamples
| H | dPR [95% CI] | dROC [95% CI] |
|---|---|---|
| 90  | **−0.00200 [−0.03701, +0.02864]  incl0** | **−0.00424 [−0.06690, +0.05988]  incl0** |
| 150 | **+0.00296 [−0.02304, +0.03133]  incl0** | **+0.02306 [−0.02672, +0.06823]  incl0** |

**Every paired CI includes 0**, on both metrics at both horizons. Point estimates are
*inconsistent in sign across H* (dROC −0.004 at H=90 vs +0.023 at H=150; dPR likewise
flips sign). No detectable improvement.

## Shape sanity — is the age→risk relation roughly monotone?
Standardized age coefficient in M-role+age: **−0.5054 (H=90), −0.4296 (H=150)** — a
strong **negative** in-sample gradient (younger → higher TJS incidence).

Positive rate by fit-set age decile (n≈1695/decile):
| decile (age range) | H=90 pos-rate | H=150 pos-rate |
|---|---|---|
| 0 (19.4–25.6) | 0.01653 | 0.02184 |
| 1 (25.6–26.7) | 0.01478 | 0.02129 |
| 2 (26.7–27.6) | 0.01177 | 0.01942 |
| 3 (27.6–28.4) | 0.01475 | 0.01888 |
| 4 (28.4–29.3) | 0.01063 | 0.01417 |
| 5 (29.3–30.3) | 0.01415 | 0.02064 |
| 6 (30.3–31.4) | 0.00295 | 0.00531 |
| 7 (31.4–32.8) | 0.00767 | 0.01180 |
| 8 (32.8–34.6) | 0.00649 | 0.01061 |
| 9 (34.6–46.3) | 0.00236 | 0.00472 |

**Roughly monotone decreasing**, Spearman(decile, pos-rate) = **−0.903 (H=90) / −0.891
(H=150)**. Local wobble (decile 5 bumps up, decile 7 recovers slightly) but no clear
nonlinearity that a single linear term misrepresents. **Per instruction, no nonlinear
terms added** — the negative linear age slope is a faithful summary.

## Proxy check — does chronic-usage proxy age?
Standardized coefficients, M-role vs M-role+age:
| feature | H=90 M-role → +age | H=150 M-role → +age |
|---|---|---|
| pc_chronic | −0.3823 → **−0.3891** | −0.3522 → **−0.3769** |
| pc_acute_dev | −0.2032 → −0.2344 | −0.1184 → −0.1288 |
| days_since_last | −0.3066 → −0.2953 | −0.2488 → −0.2481 |
| start_share | +0.2864 → +0.2793 | +0.2763 → +0.2768 |
| month | −0.1025 → −0.0727 | −0.2804 → −0.2642 |
| **age** | — → **−0.5054** | — → **−0.4296** |

Adding age **does not shrink** the workload/role coefficients toward zero — pc_chronic,
days_since_last, and start_share are essentially unchanged (pc_chronic even nudges
slightly *more* negative). Age enters with its **own** large negative coefficient. So in
this cohort the chronic-usage signal is **not a hidden proxy for age**: the two carry
largely orthogonal in-sample information, and controlling for one does not dissolve the
other.

## Verdict
**No, age does not add detectable signal on top of M-role at ~52–56 test events.** Both
paired dROC and dPR CIs include 0 at both horizons, and the point estimates flip sign
across H (H=90 slightly negative, H=150 slightly positive). This is the same power-limited
regime as the role work: with 52–56 events the paired bootstrap cannot resolve deltas of
±0.02–0.03 ROC.

Two honest qualifications that keep this from being a null-and-done:
1. **Age is a real univariate risk gradient** — monotone-decreasing incidence with age
   (younger pitchers carry ~3–4× the TJS rate of the oldest decile), sign consistent
   with the survivorship expectation. It just doesn't survive as an *incremental,
   held-out* ranking gain once workload+role are in.
2. **Age is not redundant with chronic-usage** — it does not absorb pc_chronic/start_share
   in the fit, so the "chronic-usage story proxies age" hypothesis is *not* supported;
   the muted incremental ROC/PR is a power ceiling, not collinearity.

The H=150 point estimates (dROC +0.023, evrec@50 15→18) lean positive and are the more
suggestive of the two horizons, but nothing clears the paired-CI bar. Treat age as a
legitimate, cheap, public covariate to *carry* for interpretability and calibration, not
as a demonstrated predictive-lift term at this sample size.

## Caveats
1. ~52 (H=90) / 56 (H=150) test events → wide CIs, dROC underpowered by design (Riley
   stable-estimation bound ~116–200 events).
2. Point-estimate sign of the age delta is horizon-dependent and near zero; do not
   over-read either sign.
3. Age is an in-sample negative gradient partly reflecting **survivorship** (pitchers who
   reach 34+ TJS-free are selected-durable), not necessarily a causal protective effect.
4. Age coef negative here refers to the *conditional* slope given workload+role; the raw
   decile table shows the same negative marginal trend.
5. Absolute probabilities uncalibrated (balanced class weights); rankings only. All models
   share the frozen split, resamples (seed 0), and pipeline.
6. Birthdate = MLB StatsAPI primary DOB; unverified against any second source, but 100%
   resolved and all ages fall in a plausible 19–46 window.

## Artifacts
- `r2_age.py` — analysis script (this run)
- `fetch_birthdates.py` — StatsAPI birthdate fetch
- `birthdates.csv` — 1252 pitcher birthdates (100% coverage)
- `r2_results.csv` — main rows, paired deltas, age-decile table, coefficients
- `R2_AGE.md` — this memo
