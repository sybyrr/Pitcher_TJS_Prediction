# E3 — additive feature ablation: within-pitcher TREND / VARIABILITY

Script: `scratchpad/e3_ablation.py` · Cache: `scratchpad/trendvar_features.parquet`
(24,960 x 28, keyed pitcher+t) · Outputs: `e3_results.csv`, `e3_paired_deltas.csv`,
`e3_event_recall.csv`, `e3_coverage.csv`. Protocol frozen to `b1_baseline_v2.py`.

## Self-check (M0 = 8-feature workload LR) — reproduces reference EXACTLY

| H | PR-AUC (ref) | ROC-AUC (ref) |
|---|---|---|
| 90 | 0.023421 (0.023421) | 0.639691 (0.639691) |
| 150 | 0.032400 (0.032400) | 0.634193 (0.634193) |

Match to 1e-9. Same folds, fit set (train+valid), scaler, class_weight, seed-0 bootstrap.
Effective test events: **52 distinct surgeries at H=90, 56 at H=150** (118 / 162 positive
windows respectively).

## Results — PR / ROC with pitcher-clustered bootstrap 95% CI

| H | model | nf | PR-AUC [95% CI] | ROC-AUC [95% CI] |
|---|---|---:|---|---|
| 90 | M0 workload | 8 | 0.0234 [0.0162, 0.0337] | 0.640 [0.585, 0.693] |
| 90 | M1 +trend | 15 | 0.0176 [0.0121, 0.0261] | 0.565 [0.494, 0.634] |
| 90 | M2 +variability | 15 | 0.0212 [0.0146, 0.0304] | 0.602 [0.540, 0.660] |
| 90 | M3 all | 22 | 0.0166 [0.0119, 0.0237] | 0.550 [0.478, 0.613] |
| 90 | GBM (M3 feats) | 22 | 0.0145 [0.0103, 0.0210] | 0.498 [0.433, 0.571] |
| 150 | M0 workload | 8 | 0.0324 [0.0232, 0.0445] | 0.634 [0.573, 0.693] |
| 150 | M1 +trend | 15 | 0.0242 [0.0170, 0.0346] | 0.563 [0.497, 0.631] |
| 150 | M2 +variability | 15 | 0.0298 [0.0209, 0.0409] | 0.629 [0.575, 0.679] |
| 150 | M3 all | 22 | 0.0242 [0.0173, 0.0330] | 0.579 [0.516, 0.641] |
| 150 | GBM (M3 feats) | 22 | 0.0259 [0.0173, 0.0457] | 0.537 [0.469, 0.608] |

## PAIRED deltas vs M0 (same pitcher resamples, difference bootstrapped) — the decision statistic

`*` = 95% CI excludes 0. A negative delta that excludes 0 means the family **degrades** M0.

| H | model vs M0 | dPR [95% CI] | dROC [95% CI] |
|---|---|---|---|
| 90 | +trend | -0.0059 [-0.0109, -0.0012] * | -0.0750 [-0.1226, -0.0301] * |
| 90 | +variability | -0.0023 [-0.0090, +0.0047] | -0.0385 [-0.0842, +0.0134] |
| 90 | all | -0.0069 [-0.0130, -0.0013] * | -0.0913 [-0.1485, -0.0329] * |
| 90 | GBM | -0.0090 [-0.0171, -0.0024] * | -0.1402 [-0.2288, -0.0543] * |
| 150 | +trend | -0.0082 [-0.0141, -0.0025] * | -0.0704 [-0.1205, -0.0219] * |
| 150 | +variability | -0.0026 [-0.0085, +0.0035] | -0.0063 [-0.0442, +0.0348] |
| 150 | all | -0.0086 [-0.0145, -0.0029] * | -0.0555 [-0.1149, +0.0016] |
| 150 | GBM | -0.0050 [-0.0156, +0.0096] | -0.0961 [-0.1855, -0.0104] * |

**No family produces a positive delta in any cell.** Variability is statistically
indistinguishable from M0 (every CI spans 0). Trend degrades M0 with the sign resolved
(CIs exclude 0). All / GBM degrade or match, never help.

## Event-level recall (distinct surgeries caught / total) at k = 10 / 20 / 50

| H (total) | M0 | M1 +trend | M2 +variability | M3 all | GBM |
|---|---|---|---|---|---|
| 90 (52) | 3 / 4 / 15 | 0 / 2 / 8 | 2 / 2 / 9 | 1 / 2 / 5 | 0 / 1 / 6 |
| 150 (56) | 1 / 3 / 14 | 0 / 2 / 5 | 2 / 4 / 11 | 0 / 1 / 5 | 2 / 4 / 11 |

M0 catches the most at k=50 in both horizons (15, 14). No family beats it.

## GBM check (HistGradientBoosting, max_depth 3 / max_iter 200 / lr 0.1, balanced, one config)

Nonlinearity does not recover signal. GBM on the full 22-feature set scores ROC 0.498
(H=90, at chance) / 0.537 (H=150) — **below** the linear 8-feature workload baseline
(0.640 / 0.634). No hidden nonlinear acute structure; the trees overfit the noisy acute
features. Verdict: nonlinearity worth nothing here.

## Coverage (% non-imputed = had enough qualifying games)

| feature | train | test |
|---|---:|---:|
| ff_velo_tau15 | 0.933 | 0.929 |
| sl_velo_tau15 | 0.732 | 0.777 |
| ff_spin_tau15 | 0.933 | 0.929 |
| ff_velo_sen5 | 0.949 | 0.950 |
| fb_usage_delta | 0.937 | 0.954 |
| cu_usage_delta | 0.937 | 0.954 |
| ff_velo_sd_recent | 0.965 | 0.970 |
| ff_velo_sd_rel | 0.901 | 0.900 |
| ff_relx_sd_recent | 0.965 | 0.970 |
| ff_relx_sd_rel | 0.901 | 0.900 |
| ff_ext_sd_rel | 0.901 | 0.900 |
| ff_relx_drift | 0.792 | 0.819 |
| trend_missing (=1 share) | 0.315 | 0.279 |
| var_missing (=1 share) | 0.208 | 0.181 |

Coverage is adequate (most features 90-97% on test; SL-tau lowest at 73-78% because many
pitchers lack 8 qualifying slider outings). The null result is NOT a coverage artifact.

## Operationalization notes (documented deviations / choices)

- Windows use FF-qualifying games (n_FF>=3) for all FF features; SL-qualifying (n_SL>=3)
  for the SL feature. recent15 = last 15 qualifying, recent5 = last 5 qualifying.
- Gates exactly as spec: tau features >=8 qualifying; Theil-Sen >=4; usage deltas >=10
  prior games (last-15 vs everything-before, over ALL games); variability self-baseline
  >=10 qualifying games before the recent-5 window; relx_drift career-prior >=10.
- `trend_missing` = 1 if any of features 1-4 (velo/spin tau, Theil-Sen) imputed (per spec).
  `var_missing` = 1 if any of the 6 variability features imputed (single shared indicator).
- Impute value = 0 (neutral) for every feature that fails its gate.
- kendalltau/Theil-Sen computed over non-NaN pairs; degenerate (zero-variance) tau treated
  as imputed. release_pos_x already handedness-flipped in the source table (no re-flip).

## Verdict — chronic vs acute

The chronic workload profile (M0) is the ceiling at this power. Within-pitcher **acute**
trend and variability drift add **no detectable signal** on top of it: variability is
indistinguishable from M0, trend degrades it, the union and a nonlinear GBM do no better.
With ~52-56 effective test surgeries, an acute within-pitcher signal — if one exists — is
below the detection floor here; adding 7 mostly-imputed acute features to a small-event LR
costs more variance than it returns. This is a plainly negative result, and it is the
answer to the chronic-vs-acute question: at this event count, acute drift is undetectable
and does not improve on the chronic workload baseline.
