# E0b — Risk-set / shutdown audit (activity floor F)

Universe: `slim_games.parquet` (regular-season, 154,207 pitcher-games, 2,103
pitchers, 2016-2023, DATA_END = 2023-10-01). Labels: `tj_live_clean.csv` (LIVE,
2,685 dated+id'd surgeries) for the risk-set/zombie/gap tasks; SNAPSHOT
`list of TJ.csv` additionally for Task 4 to reconcile with X1. Grid: t = 1st of
Apr..Sep, 2017-2023 = **42 dates**. Risk set at t per floor F: `>=20 games
strictly before t` AND `days_since_last (dsl) = (t - last game before t).days
<= F`. Pre-recency pool (>=20 games only) = 36,159 (pitcher, t) windows.

**Reconciliation with X1 is exact**: SNAPSHOT catchability F=INF gives 53@H150 /
56@H365 (= X1 "no_active"); F=30 gives 39@H150 / 54@H365 (= X1 "current");
14 censored = F=INF@150 − F=30@150. Grid (Apr-Sep 2017-23) matches windows.npz.

---

## Task 1 — days_since_last distribution in the risk set (pooled over 42 t)

| F | n windows | median dsl | p90 dsl | %>180d | %>365d |
|--:|----------:|-----------:|--------:|-------:|-------:|
| 365 | 24,960 | 11 | 283 | 33.03% | 0 (capped) |
| 548 | 26,329 | 17 | 322 | 36.51% | 5.20% |
| 730 | 29,208 | 40 | 547 | 42.77% | 14.54% |
| INF | 36,159 | **187** | **1324** | 53.77% | 30.97% |

The F=INF pathology is stark: **half of all "at-risk" windows are pitchers who
have not thrown a regular-season pitch in >6 months** (median dsl 187d), p90 =
1,324d (3.6 yr), and 31% are >1 year dark. Once a pitcher clears 20 games he
never leaves the F=INF risk set — retirement is indistinguishable from
mid-career. Even F=365 admits a stale tail (33% of windows >180d), but that tail
is dominated by returning/seasonal pitchers, not retirees (see Task 2).

## Task 2 — ZOMBIE windows (retirement pollution of the negative class)

Restriction: **t <= 2021-09-01** (30 grid dates, 2017-04..2021-09) so every
window has >=2 yr of lookahead through 2023. Zombie = negative at B=0 (no
surgery in (t, t+H]) AND pitcher has NO regular-season game after t.

| F | H | negatives | zombies | zombie share of negatives |
|--:|--:|----------:|--------:|--------------------------:|
| 365 | 150 | 16,697 | 2,227 | **13.34%** |
| 365 | 365 | 16,394 | 2,224 | 13.57% |
| 548 | 150 | 17,486 | 2,704 | 15.46% |
| 730 | 150 | 19,258 | 3,983 | **20.68%** |
| 730 | 365 | 18,953 | 3,980 | 21.00% |
| INF | 150 | 21,995 | 6,463 | **29.38%** |
| INF | 365 | 21,689 | 6,460 | 29.78% |

Under F=INF nearly **3 of every 10 negatives are pitchers who never appear
again** — pure retirement labelled "no surgery." Share barely moves with H
(zombies have no future, so almost none convert to positive). Dropping the floor
to INF more than doubles the zombie share vs F=365 (13.3% -> 29.4%).

## Task 3 — Return-gap evidence for a "still at risk" cutoff

Consecutive-game gaps (1,922 pitchers with >=2 games, 152,104 gaps):
**p95 = 20d, p99 = 260d, max = 2,559d** (median 3, mean 12.7). 99% of all
in-career gaps close within 260 days — normal in-season/off-season continuation
sits far below a 365-day line.

>365-day silences, split into bridged (a later game exists) vs unbridged
(trailing: last game -> DATA_END > 365d):

| | count | share | TJ inside interval |
|---|--:|--:|--:|
| bridged (pitcher returns) | 552 | **31.7%** | 123 (**22.3%**) — rehab returns |
| unbridged (never returns) | 1,191 | 68.3% | 77 (6.5%) had a later TJ |
| total >365d silences | 1,743 | 100% | |

Only ~1 in 3 of >365-day silences is ever bridged; the other two-thirds are
retirement/exit. Bridged >365d gaps have **median length 632 days** (p75 760,
p90 1,110) — squarely the TJ-rehab band (14-20 mo), and 22% contain a TJ date.
Return after a year-plus of silence is rare and disproportionately a surgical
rehab; permanent exit is the modal outcome. This is the empirical basis for
capping "still at risk" near ~1 year rather than F=INF.

## Task 4 — Censored-surgery recovery (catchability, B=0)

n catchable = >=1 grid t with (>=20 games before t) AND (dsl<=F) AND surgery in
(t, t+H]. Denominator = 2022-2023 has-history surgeries (SNAPSHOT 81 = X1;
LIVE 85 = honest label set).

SNAPSHOT (out of 81), with F=30 OLD-rule reference:

| F | H=90 | H=150 | H=365 |
|--:|-----:|------:|------:|
| 30 (OLD) | 36 (44.4%) | 39 (48.1%) | 54 (66.7%) |
| 365 | 49 (60.5%) | **52 (64.2%)** | 55 (67.9%) |
| 548 | 49 (60.5%) | 52 (64.2%) | 55 (67.9%) |
| 730 | 49 (60.5%) | 52 (64.2%) | 56 (69.1%) |
| INF | 50 (61.7%) | **53 (65.4%)** | 56 (69.1%) |

LIVE (out of 85): F=365 -> 52/56/59; F=730 -> 52/56/60; F=INF -> 53/57/60;
F=30 -> 39/43/58. Same shape, +~4 surgeries from the completer label sheet.

**The 14 informatively-censored surgeries (X1 = F=INF@150 − F=30@150)** —
recovery on that fixed set, per F × H:

| F | H=90 | H=150 | H=365 |
|--:|-----:|------:|------:|
| 365 | 12 | **13** | 13 |
| 730 | 12 | 13 | 14 |
| INF | 13 | **14** | 14 |

At H=150, F=365 already recovers 13 of the 14; only Trevor Rosenthal (977-day
pre-surgery gap) needs F=INF. Their pre-surgery gaps range 13-977d (median ~200):
mostly multi-month go-dark shutdowns that the 30-day recency gate discarded.

## Task 5 — FLOOR RECOMMENDATION: **F = 365 days**

Efficiency frontier (H=150): catchability climbs 39 (F=30) -> 52 (F=365) then
plateaus (52 at 548/730, 53 at INF); zombie pollution climbs monotonically
13.3% (365) -> 15.5% (548) -> 20.7% (730) -> 29.4% (INF).

1. **F=365 closes 93% of the recency leak X1 flagged**: 52/81 vs OLD 39/81 at
   H=150, recovering 13 of 14 censored surgeries.
2. **At <half the retirement pollution of F=INF** (13.3% vs 29.4% zombie
   negatives). Beyond 365 the catchability gain is 0-1 surgery per floor step
   while zombie share keeps rising — a strictly worse trade.
3. **365 clears the p99 of normal in-career gaps (260d)** so essentially every
   legitimate seasonal continuation is retained; the excluded tail is the
   retirement mass (F=INF median dsl = 187d, p90 = 1,324d).
4. **The single surgery F=365 forfeits (Rosenthal, 977d dark) is
   indistinguishable from a retired pitcher**; buying it via F=INF imports 6,463
   zombie windows.
5. **Pair with H=365**: catchability -> 55/81 (67.9%); the off-season /
   grid-boundary censoring is fixed by the horizon, not by loosening the floor.

**Modeling requirement (report note):** at F>30 the `days_since_last` feature
becomes informative and must be included as a continuous covariate — under F=365
it spans 0-365 and 33% of the risk set is >180d. X2's structural finding was that
the OLD rule truncated dsl at 30, censoring exactly the go-dark signal; the floor
should gate eligibility while dsl is kept as a feature, not thrown away.

**Alternative — F=548** if a later phase targets re-injury of *returning*
pitchers (bridged-return median = 632d, so 548 keeps most rehabbers in the risk
set). It costs +2pp zombie over F=365 and gains **0** index-surgery catches;
justified only if the prediction target expands to post-return re-tears.

## Caveats

- Zombie/never-return counts inflate near DATA_END is avoided by the t<=2021-09
  restriction, but "no game after t through 2023" still labels as retired some
  pitchers who merely dropped out of the FF-tracked starter pool (minors, other
  orgs). It is an upper bound on true retirement, but the F-ranking is monotone
  regardless.
- Catchability denominators differ by label source (SNAPSHOT 81 is incomplete
  for 2022-23; LIVE 85 is the honest set per E0a). Percentages, not counts, are
  the stable comparison across F.
- dsl can be 0 in principle (game on t); with t on the 1st and games rarely on
  the 1st this is negligible and does not affect the >180/>365 tails.

Artifacts (scratchpad): e0b_audit.py, e0b_censored14.py, e0b_f548.py,
e0b_task1_dsl.csv, e0b_task2_zombies.csv, e0b_task4_catch.csv.
