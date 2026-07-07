# M0' curated 5-feature model vs M0 8-feature reference

Protocol: adapted verbatim from `results/phase26/scripts/b1_baseline_v2.py`
(StandardScaler + LogisticRegression class_weight=balanced, max_iter=2000; fit on
fold_main train+valid; eval on test; 1000 pitcher-clustered bootstrap resamples,
seed 0, shared across models for paired CIs). Cells: H∈{90,150}, B=0, fold_main.

- **M0**  (8 feats): pc_30, pc_90, ng_30, ng_90, acwr, days_since_last, vel_trend, month
- **M0'** (5 feats): pc_30, pc_90, days_since_last, vel_trend, month  — drops ng_30, ng_90, acwr

## Reproduction check (M0 = b1 'workload')
| H | M0 PR-AUC | ref | M0 ROC-AUC | ref | |
|---|---|---|---|---|---|
| 90  | 0.023421 | 0.023421 | 0.639691 | 0.639691 | MATCH |
| 150 | 0.032400 | 0.032400 | 0.634193 | 0.634193 | MATCH |

## Main table (test set; CI = 95% pitcher-clustered bootstrap, 1000/1000 valid)
| H | model | PR-AUC [95% CI] | ROC-AUC [95% CI] | base rate | n_pos |
|---|---|---|---|---|---|
| 90  | M0  | 0.023421 [0.0162, 0.0337] | 0.639691 [0.5848, 0.6929] | 0.01473 | 118 |
| 90  | M0' | 0.022700 [0.0157, 0.0330] | 0.615557 [0.5538, 0.6760] | 0.01473 | 118 |
| 150 | M0  | 0.032400 [0.0232, 0.0445] | 0.634193 [0.5734, 0.6932] | 0.02022 | 162 |
| 150 | M0' | 0.030203 [0.0220, 0.0412] | 0.619131 [0.5622, 0.6745] | 0.02022 | 162 |

## Paired deltas (M0' − M0, same resamples)
| H | dPR point [95% CI] | dROC point [95% CI] |
|---|---|---|
| 90  | −0.000721 [−0.00376, +0.00211] | −0.024134 [−0.06095, +0.01407] |
| 150 | −0.002197 [−0.00599, +0.00111] | −0.015063 [−0.03966, +0.01516] |

Both dROC and both dPR CIs include 0 → **M0' is not detectably worse** than M0 on
either metric, at either horizon. Point estimates favor M0 by a small margin
(ROC lower by 0.015–0.024).

## Event recall (distinct surgeries caught at top-k per decision date)
| H | model | @10 | @20 | @50 | total |
|---|---|---|---|---|---|
| 90  | M0  | 3 | 4  | 15 | 52 |
| 90  | M0' | 5 | 10 | 20 | 52 |
| 150 | M0  | 1 | 3  | 14 | 56 |
| 150 | M0' | 2 | 3  | 13 | 56 |

On the operational metric M0' is **equal-or-better**: clearly better at H=90 (20 vs
15 @50), a wash at H=150 (13 vs 14 @50). Absolute recall is low for both (top-50
of ~8k windows).

## M0' standardized coefficients (interpretable — score = higher predicted risk)
| feature | H=90 | H=150 | directional read |
|---|---|---|---|
| pc_30 | −0.3338 | −0.1543 | higher recent 30d pitch count → lower risk (declining workload precedes surgery; strongest single signal) |
| pc_90 | +0.0900 | −0.0378 | ~0, sign flips across H — variance shared with pc_30 (r=0.86), not independently interpretable |
| days_since_last | −0.2412 | −0.1808 | longer layoff → lower risk (counterintuitive; confounded, dsl~pc_30 r=−0.60) |
| vel_trend | −0.0201 | +0.0042 | ≈0 — velocity-loss signal is negligible in this linear model |
| month | −0.1473 | −0.3244 | later in season → lower risk of surgery-within-H (seasonal; strongest at H=150) |

Calibration (M0'): H=90 slope 1.601 / intercept −4.252; H=150 slope 1.178 /
intercept −3.919. Slopes >1 (scores slightly under-dispersed) but H=150 near-ideal.

## VIF — M0' 5-feature set (fit set n=16949)
| feature | VIF |
|---|---|
| pc_90 | 4.47 |
| pc_30 | 4.35 |
| days_since_last | 1.64 |
| month | 1.25 |
| vel_trend | 1.03 |

max VIF **4.47** (cond number 4.55). Dropping ng/acwr cut VIF from the M0 range
8.6–9.7 to 4.47 — roughly halved — but the pc_30~pc_90 pair (r=0.86) remains, so
it does **not** clear 3.5. To reach <3.5 one of pc_30/pc_90 would also have to go.

## Blackout sweep (M0', H=150, B∈{0,30,60,90}) — flatness confirmed
| B | n_pos | base rate | PR-AUC [95% CI] | lift | ROC-AUC [95% CI] |
|---|---|---|---|---|---|
| 0  | 162 | 0.02022 | 0.03020 [0.0220, 0.0412] | 1.49× | 0.61913 [0.5622, 0.6745] |
| 30 | 119 | 0.01485 | 0.02256 [0.0159, 0.0316] | 1.52× | 0.62144 [0.5578, 0.6842] |
| 60 | 78  | 0.00974 | 0.01511 [0.0102, 0.0225] | 1.55× | 0.63414 [0.5586, 0.7087] |
| 90 | 44  | 0.00549 | 0.00865 [0.0054, 0.0142] | 1.58× | 0.64010 [0.5393, 0.7337] |

ROC flat/slightly rising across B (0.619→0.640); PR falls with the base rate but
lift holds ~1.5×. Same flatness M0 showed → not a fill/leakage artifact of M0.

## Success criteria
- **(a) M0 reproduction exact — PASS** (both cells match ref to 1e-6).
- **(b) paired dROC CI includes 0 (both H) — PASS** (H=90 [−0.061,+0.014]; H=150 [−0.040,+0.015]).
- **(c) max VIF(M0') < 3.5 — FAIL** (max 4.47; pc_30~pc_90 r=0.86 not resolved by dropping ng/acwr).

**Verdict.** M0' is a defensible simplification: statistically indistinguishable
from M0 (dROC/dPR CIs cover 0), equal-or-better event recall, decent calibration,
and now yields readable coefficients (workload-decline + seasonal signals). The
one miss is collinearity — keeping both pc_30 and pc_90 leaves VIF ~4.4, so
criterion (c) is not met. A stricter curation would drop one of the two windows.

---

# M0'' — curated-orthogonal (reparametrized workload pair)

Script `m0doubleprime.py` (append-only to `m0prime_results.csv`, section
`m0doubleprime*`). Same frozen protocol; only the workload pair is
**reparametrized** — same information, decorrelated:

- **pc_chronic**   = pc_90 / 90.0            — trailing-90d daily pitch rate (chronic level)
- **pc_acute_dev** = pc_30/30.0 − pc_90/90.0 — recent-30d rate minus chronic rate
  (positive = ramping up, negative = usage decline)
- **M0''** = {pc_chronic, pc_acute_dev, days_since_last, vel_trend, month}

The map (pc_30, pc_90) → (pc_chronic, pc_acute_dev) has matrix
[[0, 1/90],[1/30, −1/90]], **det = −1/2700 ≠ 0**, so it is invertible: M0'' and M0'
span the *same affine subspace*. An unpenalized LR would give identical
predictions; sklearn's L2 penalty (C=1) acts on the standardized coefficients, so
only the penalty geometry differs → predictions differ by a whisper.

## Main table (test set; CI = 95% pitcher-clustered bootstrap, 1000/1000 valid)
| H | model | PR-AUC [95% CI] | ROC-AUC [95% CI] | base rate | n_pos |
|---|---|---|---|---|---|
| 90  | M0'' | 0.022712 [0.0157, 0.0330] | 0.615687 [0.5540, 0.6760] | 0.01473 | 118 |
| 90  | (M0' ref) | 0.022700 [0.0157, 0.0330] | 0.615557 [0.5538, 0.6760] | 0.01473 | 118 |
| 150 | M0'' | 0.030076 [0.0219, 0.0409] | 0.618917 [0.5622, 0.6746] | 0.02022 | 162 |
| 150 | (M0' ref) | 0.030203 [0.0220, 0.0412] | 0.619131 [0.5622, 0.6745] | 0.02022 | 162 |

## Paired deltas (same 1000 resamples)
| H | vs | dPR point [95% CI] | dROC point [95% CI] | |dROC|>0.01? |
|---|---|---|---|---|
| 90  | M0'' − M0' | +0.000012 [+0.000003, +0.000025] | +0.000130 [+0.000032, +0.000231] | no |
| 90  | M0'' − M0  | −0.000709 [−0.00374, +0.00213] | −0.024004 [−0.06074, +0.01412] | (vs M0, not the flag) |
| 150 | M0'' − M0' | −0.000127 [−0.000410, +0.000024] | −0.000214 [−0.000650, +0.000121] | no |
| 150 | M0'' − M0  | −0.002324 [−0.00604, +0.00092] | −0.015277 [−0.03976, +0.01481] | (vs M0, not the flag) |

Deltas **vs M0' are ~1e-4** on both metrics — three orders below the 0.01 flag, so
the flag does **not** fire at either H. Deltas vs M0 mirror M0' almost exactly
(dROC −0.024 / −0.015, CIs cover 0): M0'' inherits M0's "not detectably worse than
M0" status. **Caveat:** the M0''−M0' CI **excludes 0 at H=90** (both dPR & dROC
strictly positive) and includes 0 at H=150. This is the expected L2-geometry
signature, not a real gap: the two score vectors are ~0.9999-collinear, so the
*difference* has minuscule resampling variance and 1000 resamples resolve a
0.0001-size shift away from zero. Effect size is negligible; statistical
resolvability ≠ meaningful difference.

## Event recall (distinct surgeries caught at top-k per decision date)
| H | model | @10 | @20 | @50 | total |
|---|---|---|---|---|---|
| 90  | M0'' | 5 | 10 | 19 | 52 |
| 90  | (M0') | 5 | 10 | 20 | 52 |
| 150 | M0'' | 2 | 3  | 13 | 56 |
| 150 | (M0') | 2 | 3  | 13 | 56 |

Identical to M0' except one fewer catch at H=90 @50 (19 vs 20) — a single ranking
flip from the L2-geometry perturbation, consistent with the ~1e-4 metric delta.

## M0'' standardized coefficients (score = higher predicted risk; pair now interpretable)
| feature | H=90 | H=150 | directional read |
|---|---|---|---|
| pc_chronic | −0.1850 | −0.1643 | higher chronic (90d) daily pitch rate → lower risk; the "declining-workload-precedes-surgery" **level** signal, now isolated |
| pc_acute_dev | −0.1718 | −0.0792 | recent-30d rate **above** 90d baseline (ramping up) → lower risk; a **drop below** baseline (usage cut) → higher risk. Nearer-term signal (stronger at H=90) |
| days_since_last | −0.2410 | −0.1799 | longer layoff → lower risk (same confounded read as M0'; dsl~pc_chronic r=−0.57) |
| vel_trend | −0.0201 | +0.0042 | ≈0 — velocity-loss signal negligible, as in M0' |
| month | −0.1475 | −0.3245 | later in season → lower surgery-within-H risk (seasonal; strongest at H=150) |

Contrast with M0': there {pc_30 (−0.33), pc_90 (+0.09/−0.04, sign-flipping)} was
jointly uninterpretable (r=0.86). The reparametrization keeps predictions
identical but rotates the pair into a **level + deviation** basis that is nearly
orthogonal (r=0.066), so both coefficients are stable in sign across H and
independently readable. Calibration (M0''): H=90 slope 1.602 / int −4.252; H=150
slope 1.178 / int −3.919 (unchanged from M0').

## VIF — M0'' 5-feature set (fit set n=16949)
| feature | VIF |
|---|---|
| days_since_last | 1.637 |
| pc_chronic | 1.620 |
| month | 1.251 |
| pc_acute_dev | 1.151 |
| vel_trend | 1.030 |

max VIF **1.637**, condition number **2.176**. Pearson r(pc_chronic,
pc_acute_dev) on the fit set = **+0.066** (was 0.86 for pc_30~pc_90). The
reparametrization dissolves the collinearity that M0' could not.

## Success criteria
- **max VIF < 3.5 — PASS** (1.637; expected ~1.5–2, confirmed).
- **not detectably different from M0' (paired CIs include 0) — MIXED/strict-FAIL:**
  H=150 CIs include 0 (pass); H=90 CIs **exclude 0** (dPR [+3e-6,+2.5e-5], dROC
  [+3e-5,+2.3e-4]). Both point |dROC| < 0.01 so the task's stated flag does not
  fire; the CI-excludes-0 is an L2-geometry artifact at negligible effect size.

**Verdict.** M0'' delivers what M0' could not: **max VIF 1.64 (< 3.5) and readable,
sign-stable workload coefficients** (chronic level vs acute deviation, r=0.066),
while keeping M0''s predictions numerically indistinguishable from M0' (deltas
~1e-4) and its "not worse than M0" status (dROC vs M0 CIs cover 0). The only
asterisk is that with ~0.9999-collinear scores the paired H=90 CI resolves the
tiny L2-penalty difference away from zero — real but practically meaningless.
M0'' is the recommended curated baseline: interpretable and collinearity-clean at
no measurable performance cost.
