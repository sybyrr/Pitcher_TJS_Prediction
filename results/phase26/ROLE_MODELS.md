# Role-aware models vs canonical M0dp — frozen protocol

Script `role_models.py` (self-contained; frozen protocol adapted **verbatim** from
`results/phase26/scripts/m0doubleprime.py`: StandardScaler + LogisticRegression
(class_weight='balanced', max_iter=2000), fit on fold_main train+valid, test 2022–23;
1000 pitcher-clustered bootstrap resamples, seed 0, **shared across models** for
paired CIs; event recall via `next_surgery_date` grouping). Results CSV:
`role_models_results.csv`.

Models (per H∈{90,150}, B=0, fold_main):
- **M0dp** (reference, 5): pc_chronic, pc_acute_dev, days_since_last, vel_trend, month
- **M2** +role additive: M0dp + `start_share` (continuous)
- **M3** role interactions: M0dp + start_share + start_share·z(pc_chronic) +
  start_share·z(pc_acute_dev) + start_share·z(days_since_last)
- **M4** within-role standardized: within-role z(pc_chronic, pc_acute_dev,
  days_since_last) + start_share + vel_trend + month

## Role proxy (leakage-free) & sanity checks
Per (pitcher,t), over `game_features_v2.total_pitches` for games **strictly before t**
in the trailing **365 days**: `start_share` = fraction with total_pitches ≥ 50 (0 if
no games); `role_sp` = 1 if start_share ≥ 0.5. (`total_pitches` verified byte-identical
to slim `pitch_count`, max abs diff 0.0.) Diagnostics reproduce the task's known values:

| diagnostic | computed | expected |
|---|---|---|
| fit-set SP share (role_sp==1) | **0.3198** | ~0.32 |
| pc_chronic median SP / RP (fit) | **4.7333 / 0.9889** | ~4.73 / ~0.99 |
| test base rate H=90 SP / RP | **0.0222 / 0.0113** | ~0.0222 / ~0.0113 |
| test base rate H=150 SP / RP | 0.0297 / 0.0159 | (SP > RP) |

## M0 reproduction (must hit anchors)
| H | M0dp ROC | anchor | Δ | M0dp PR | anchor | Δ |
|---|---|---|---|---|---|---|
| 90  | 0.615687 | 0.615557 | +1.3e-4 | 0.022712 | 0.022700 | +1.2e-5 |
| 150 | 0.618917 | 0.619131 | −2.1e-4 | 0.030076 | 0.030203 | −1.3e-4 |

MATCH to ~1e-4 ROC / ~1e-5 PR. **Note:** M0dp is defined with pc_chronic/pc_acute_dev
= the project's canonical **M0''** (double-prime). My reproduction is byte-exact to the
frozen `m0doubleprime_results.csv` M0dprime row. The task's anchors (0.615557 /
0.619131) are the **M0'** (single-prime, pc_30/pc_90 basis) values from `M0PRIME.md`;
M0''−M0' differs only by the documented L2-penalty-geometry whisker (dROC ≈ +1.3e-4),
so the anchor is matched within that whisker.

## Main table (test set; 95% pitcher-clustered bootstrap, 1000/1000 valid)
### H=90 (52 distinct test surgeries)
| model | PR-AUC [95% CI] | ROC-AUC [95% CI] | evrec @10/20/50 |
|---|---|---|---|
| M0dp | 0.02271 [0.01567, 0.03299] | 0.61569 [0.55396, 0.67603] | 5 / 10 / 19 |
| M2 +role add | 0.03597 [0.01989, 0.07080] | 0.64368 [0.57643, 0.70933] | 6 / 7 / 13 |
| M3 role inter | 0.03268 [0.01882, 0.06071] | 0.64291 [0.57509, 0.70661] | 4 / 8 / 15 |
| M4 within-role | 0.03194 [0.01871, 0.05933] | 0.64801 [0.58299, 0.70617] | 3 / 10 / 16 |

### H=150 (56 distinct test surgeries)
| model | PR-AUC [95% CI] | ROC-AUC [95% CI] | evrec @10/20/50 |
|---|---|---|---|
| M0dp | 0.03008 [0.02190, 0.04094] | 0.61892 [0.56218, 0.67460] | 2 / 3 / 13 |
| M2 +role add | 0.04700 [0.02960, 0.07862] | 0.64377 [0.57859, 0.70722] | 8 / 11 / 15 |
| M3 role inter | 0.05333 [0.03136, 0.09091] | 0.64547 [0.57470, 0.71308] | 9 / 11 / 16 |
| M4 within-role | 0.05022 [0.02977, 0.08403] | 0.64693 [0.58032, 0.71178] | 8 / 11 / 16 |

## Paired deltas vs M0dp (same 1000 resamples; ↑EXCL0 = CI excludes 0 upward)
### H=90
| model | dPR [95% CI] | dROC [95% CI] |
|---|---|---|
| M2 | **+0.01326 [+0.00020, +0.04256]  ↑EXCL0** | +0.02799 [−0.02702, +0.08522]  incl0 |
| M3 | +0.00997 [−0.00012, +0.03350]  incl0 | +0.02722 [−0.03364, +0.08971]  incl0 |
| M4 | +0.00923 [−0.00004, +0.03199]  incl0 | +0.03232 [−0.02586, +0.08862]  incl0 |

### H=150
| model | dPR [95% CI] | dROC [95% CI] |
|---|---|---|
| M2 | **+0.01692 [+0.00229, +0.04265]  ↑EXCL0** | +0.02486 [−0.01778, +0.07070]  incl0 |
| M3 | **+0.02325 [+0.00398, +0.05616]  ↑EXCL0** | +0.02656 [−0.02529, +0.07938]  incl0 |
| M4 | **+0.02014 [+0.00281, +0.04863]  ↑EXCL0** | +0.02801 [−0.02274, +0.07941]  incl0 |

**dROC: no role form's CI excludes 0 at either H** (all include 0; all point estimates
positive, +0.025…+0.032). **dPR: M2 excludes 0 upward at both H; M3 & M4 exclude 0 at
H=150 and graze 0 at H=90** (lower bounds −1.2e-4 / −3.9e-5, i.e. essentially at zero).

## M2 standardized coefficients (Simpson structure)
| feature | H=90 | H=150 | M0dp-alone (ref) |
|---|---|---|---|
| **start_share** | **+0.2864** | **+0.2763** | — (not in M0dp) |
| **pc_chronic** | **−0.3823** | **−0.3522** | −0.185 / −0.164 |
| pc_acute_dev | −0.2032 | −0.1184 | −0.172 / −0.079 |
| days_since_last | −0.3066 | −0.2488 | −0.241 / −0.180 |
| vel_trend | −0.0233 | +0.0003 | ≈0 |
| month | −0.1025 | −0.2804 | −0.147 / −0.324 |

**Simpson structure confirmed by fitted signs.** (i) `start_share` coefficient is
**positive** (+0.28) at both H — the between-role gradient: starters carry higher TJS
risk, matching the raw base rates (SP 0.0222 > RP 0.0113 at H=90; 0.0297 > 0.0159 at
H=150). (ii) Once role is controlled, `pc_chronic` becomes **more negative** — it
roughly doubles from −0.185 (M0dp alone) to −0.382 at H=90 (−0.164→−0.352 at H=150).
In M0dp alone the chronic-workload coefficient was a muted mixture of a positive
between-role component (SP have both high pc_chronic and high risk) and a negative
within-role component that partly cancelled it; adding `start_share` absorbs the
positive between-role prevalence, exposing the **negative within-role** usage→risk
slope ("declining/lower recent workload precedes surgery"). This is textbook
confounding/suppression: marginal slope (−0.185) and role-conditional slope (−0.382)
differ in sign-magnitude because role is positively associated with both usage and risk.

## Verdict — does any role form DETECTABLY beat M0dp (paired CI excludes 0 upward)?
**Metric- and horizon-dependent, not a clean win at ~52–56 events.**
- **ROC-AUC: NO** — every dROC CI includes 0 at both H, despite consistently positive
  point estimates (+0.025…+0.032). ROC is underpowered here (Riley stable-estimation
  bound ~116–200 events; we have 52/56).
- **PR-AUC: YES, partially** — the simplest form **M2 (+start_share) detectably beats
  M0dp at both horizons** (dPR CI excludes 0 upward); M3 & M4 detectably beat at H=150
  but graze 0 at H=90.
- **Event recall:** improved at **H=150** (M0dp 2/3/13 → role 8–9 / 11 / 15–16 —
  a large @10/@20 gain), but **not at H=90** (M0dp @50=19 is the best; role forms 13–16).

**Point-estimate directions:** all role forms shift PR-AUC, ROC-AUC, and (at H=150)
event recall **upward**; the magnitudes are meaningful-looking (ROC +~0.03, PR roughly
+50–70% relative) but only PR clears the paired-CI bar, and only cleanly for M2.

**Interpretation caveat (why it's PR-only):** the gain is largely the **between-role
prevalence gradient** — starters have ~2× the TJS base rate of relievers, and
`start_share` is essentially a base-rate stratifier. PR-AUC is highly sensitive to
ranking the higher-prevalence subgroup up, so it registers the effect; ROC (rank over
all pairs) does not resolve it at this power. This is a "starters get more TJS"
population fact, **not** new within-pitcher early-warning signal — the within-role
workload slope stays negative, consistent with Phase 2.6's "chronic risk profile, no
acute countdown." **Interactions (M3) and within-role rescaling (M4) add nothing
detectable over the additive M2** (overlapping CIs; at H=90 they actually graze 0),
so the additive `start_share` captures essentially all the role information.

## Caveats
1. ~52 (H=90) / 56 (H=150) test events → wide CIs, low power; dROC underpowered by design.
2. PR-detection is thin at H=90 (M2 dPR lower bound +0.0002); PR-AUC bootstrap is
   high-variance/right-skewed at ~1.5–2% base rate.
3. Role gain ≈ between-role prevalence (SP base rate ≈ 2× RP), captured by start_share;
   the within-role usage slope remains negative (no acute countdown recovered).
4. Role proxy is a design choice: 365-day start_share, SP/RP boundary at start_share≥0.5,
   game-start threshold total_pitches≥50.
5. Absolute probabilities uncalibrated (balanced class weights); rankings only.
6. All models share the frozen split, resamples (seed 0), and pipeline; M0dp reproduces
   the frozen M0'' row exactly.
