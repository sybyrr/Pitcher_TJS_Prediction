# SUPPLY SUMMARY — (H, B, F) horizon/blackout/floor decision grid

Counting only (no features, no models). Label source = **live** Roegele sheet `tj_live_clean.csv` (782 in-universe distinct surgeries), matched to Statcast pitchers by mlbamid. Game universe = `slim_games.parquet` (154,207 games, 2,103 pitchers, 2016-2023). Decision grid t = 1st of Apr..Sep x 2017-2023 (42 dates). Candidate windows (>=20 games strictly before t) = 36,159.

Full grid: `supply_grid.csv` (1,056 rows = 2 foldsets x 11 (H,B) x 4 F x 2 ends x 2 embargo x 3 folds). Raw no-embargo/no-boundary reference: `supply_raw_noembargo_noboundary.csv`.

**Read F as a string** — pandas coerces the literal `INF` to float `inf` and `30/365/730` to floats on naive read; use `dtype={'F':str}`.

## Sanity checks (all PASS)

| check | result |
|---|---|
| (H=150,B=0,F=30,no embargo,no boundary) vs known 102/31/84 | train_pos=**102** (=102 exact), valid_pos=**31** (=31 exact), test_pos=**95** (vs 84; live sheet adds 2022-23 surgeries). windows 7408/2180/4540 vs ref 7404/2179/4540 |
| positive windows monotone non-decreasing in H (fixed B,F,fold) | PASS |
| positive windows non-increasing in B (fixed H,F,fold) | PASS |

## DECISION TABLE — main folds, STRICT EMBARGO + BOUNDARY, E0a end (2024-12-31)

Columns: **P** = positive windows, **S** = distinct surgeries covered, trP/vaP/teP = train/valid/test. teBR = test base rate (teP / test windows).

### F = 365
| H | B | trP | vaP | teP | trS | vaS | teS | teBR |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 90 | 0 | 138 | 35 | 118 | 60 | 17 | 52 | 0.0147 |
| 90 | 30 | 87 | 23 | 75 | 51 | 13 | 44 | 0.0094 |
| 90 | 60 | 40 | 11 | 34 | 40 | 11 | 34 | 0.0042 |
| 150 | 0 | 205 | 47 | 162 | 62 | 17 | 56 | 0.0202 |
| 150 | 30 | 154 | 35 | 119 | 55 | 13 | 48 | 0.0149 |
| 150 | 60 | 107 | 23 | 78 | 47 | 11 | 38 | 0.0097 |
| 150 | 90 | 67 | 12 | 44 | 41 | 8 | 29 | 0.0055 |
| 365 | 0 | 294 | 15 | 346 | 82 | 15 | 81 | 0.0432 |
| 365 | 30 | 258 | 12 | 303 | 78 | 12 | 77 | 0.0378 |
| 365 | 60 | 225 | 10 | 262 | 73 | 10 | 74 | 0.0327 |
| 365 | 90 | 198 | 7 | 228 | 70 | 7 | 67 | 0.0285 |

### F = INF (no recency floor)
| H | B | trP | vaP | teP | trS | vaS | teS | teBR |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 90 | 0 | 143 | 38 | 121 | 60 | 19 | 53 | 0.0087 |
| 90 | 30 | 91 | 25 | 78 | 53 | 14 | 45 | 0.0056 |
| 90 | 60 | 42 | 12 | 36 | 42 | 12 | 36 | 0.0026 |
| 150 | 0 | 212 | 53 | 167 | 62 | 19 | 57 | 0.0120 |
| 150 | 30 | 160 | 40 | 124 | 55 | 14 | 49 | 0.0089 |
| 150 | 60 | 111 | 27 | 82 | 47 | 12 | 40 | 0.0059 |
| 150 | 90 | 69 | 15 | 46 | 41 | 9 | 30 | 0.0033 |
| 365 | 0 | 294 | 18 | 364 | 82 | 18 | 83 | 0.0262 |
| 365 | 30 | 258 | 14 | 321 | 78 | 14 | 79 | 0.0231 |
| 365 | 60 | 225 | 12 | 279 | 73 | 12 | 77 | 0.0201 |
| 365 | 90 | 198 | 9 | 243 | 70 | 9 | 70 | 0.0175 |

## Same table under CONSERVATIVE end (2023-12-31) — boundary bites test hard at H=365

### F = 365
| H | B | trP | vaP | teP | trS | vaS | teS | teBR |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 90 | 0 | 138 | 35 | 118 | 60 | 17 | 52 | 0.0147 |
| 90 | 30 | 87 | 23 | 75 | 51 | 13 | 44 | 0.0094 |
| 90 | 60 | 40 | 11 | 34 | 40 | 11 | 34 | 0.0042 |
| 150 | 0 | 205 | 47 | 156 | 62 | 17 | 56 | 0.0212 |
| 150 | 30 | 154 | 35 | 114 | 55 | 13 | 48 | 0.0155 |
| 150 | 60 | 107 | 23 | 76 | 47 | 11 | 38 | 0.0103 |
| 150 | 90 | 67 | 12 | 43 | 41 | 8 | 29 | 0.0059 |
| 365 | 0 | 294 | 15 | 173 | 82 | 15 | 50 | 0.0422 |
| 365 | 30 | 258 | 12 | 152 | 78 | 12 | 46 | 0.0371 |
| 365 | 60 | 225 | 10 | 133 | 73 | 10 | 44 | 0.0324 |
| 365 | 90 | 198 | 7 | 115 | 70 | 7 | 37 | 0.0280 |

### F = INF
| H | B | trP | vaP | teP | trS | vaS | teS | teBR |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 90 | 0 | 143 | 38 | 121 | 60 | 19 | 53 | 0.0087 |
| 90 | 30 | 91 | 25 | 78 | 53 | 14 | 45 | 0.0056 |
| 90 | 60 | 42 | 12 | 36 | 42 | 12 | 36 | 0.0026 |
| 150 | 0 | 212 | 53 | 161 | 62 | 19 | 57 | 0.0127 |
| 150 | 30 | 160 | 40 | 119 | 55 | 14 | 49 | 0.0094 |
| 150 | 60 | 111 | 27 | 80 | 47 | 12 | 40 | 0.0063 |
| 150 | 90 | 69 | 15 | 45 | 41 | 9 | 30 | 0.0036 |
| 365 | 0 | 294 | 18 | 181 | 82 | 18 | 52 | 0.0272 |
| 365 | 30 | 258 | 14 | 160 | 78 | 14 | 48 | 0.0240 |
| 365 | 60 | 225 | 12 | 141 | 73 | 12 | 46 | 0.0212 |
| 365 | 90 | 198 | 9 | 123 | 70 | 9 | 39 | 0.0185 |

## Embargo impact (F=INF, B=0, main; end=E0a) — train/valid pos BEFORE -> AFTER strict embargo

| H | train pos (win) | valid pos (win) |
|--:|---|---|
| 90 | 143->143 (16434->16434) | 38->38 (5826->5826) |
| 150 | 212->212 (16434->16434) | 53->53 (5826->5826) |
| 365 | 451->294 (16434->12186) | 120->18 (5826->912) |

Embargo is free at H=90 and H=150 (labels resolve before the next fold opens); at H=365 it removes 35% of train positives and **collapses main valid 120->18** (only the Apr-2021 decision date survives t+365<=2022-04-01).

## Boundary impact (F=INF, B=0, main test) — CONS(2023-12-31) vs E0a(2024-12-31)

| H | CONS teP / teS | E0a teP / teS |
|--:|---|---|
| 90 | 121 / 53 | 121 / 53 |
| 150 | 161 / 57 | 167 / 57 |
| 365 | 181 / 52 | 364 / 83 |

At H=90 the two ends are identical (t+90 resolves before 2023-12-31 for every test date). At H=365 the conservative end **drops all of 2023's test windows** (t+365 > 2023-12-31 once t>=Apr 2023): test positives 364->181, distinct surgeries 83->52. The E0a end (2024-12-31) is what makes a 365-day test set whole.

## ALT FOLDS (train 2017-19 / valid 2020-21 / test 2022-23) — rescue valid at H=365

| H | MAIN valid P/S/win | ALT valid P/S/win | ALT train P/win |
|--:|---|---|---|
| 90 | 38/19/5826 | 86/37/10914 | 95/11346 |
| 150 | 53/19/5826 | 132/39/10914 | 133/11346 |
| 365 | 18/18/912 | 202/45/6000 | 187/7461 |

Only H=365 needs the alt split: it lifts strict-embargo valid from **18->202 positives** (surg 18->45) while keeping train strong (187 pos). At H=90/150 main valid is already healthy (38/53), so the alt split is optional there.

## Per-H cost: naive vs strict embargo + E0a boundary (F=INF, B=0, main)

| H | naive tr/va/te pos (te surg) | post tr/va/te pos (te surg) | one-line cost |
|--:|---|---|---|
| 90 | 143/38/121 (53) | 143/38/121 (53) | embargo & E0a boundary both cost ZERO; supply fully robust. |
| 150 | 212/53/167 (57) | 212/53/167 (57) | embargo & E0a boundary cost ZERO (conservative end trims test ~6 windows only). |
| 365 | 451/120/364 (83) | 294/18/364 (83) | embargo cuts train -157 & collapses valid (120->18, use alt folds); E0a boundary costs test 0, but CONSERVATIVE end would cut test 364->181 & surg 83->52. |

## Viability & recommendation

Rule: after all costs (strict embargo on train, boundary on test), a cell is **viable** iff train positives >= ~80 AND test distinct surgeries >= ~30.

- **Shortest viable H = 90 days**, but ONLY with the recency floor relaxed to F in {365, 730, INF}. At H=90, F=365, B=0 (E0a end): train_pos=138, test_surg=52, valid_pos=35. Embargo and boundary each cost 0 here, so H=90 is the most cost-robust horizon. It fails at F=30 (train_pos<80: the tight OLD recency gate starves a 90-day horizon).

- If the tight F=30 recency gate is mandatory, the shortest viable H = **150** (F=30, B=0: train_pos=102, test_surg=43).

- H=365 maximizes supply (train ~294, test_surg 81-83) but needs the E0a end AND the alt fold split to keep a usable valid set; under the conservative end it loses half the test set.

- Blackout B: B=0 and B=30 stay viable across H>=90; B>=60 erodes test surgeries toward the 30-floor at H=90/150.
