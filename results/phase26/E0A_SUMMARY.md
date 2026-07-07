# E0a — Label-date audit: snapshot vs LIVE Roegele TJ list

Live sheet pulled 2026-07-07 via `export?format=csv` (spreadsheet id
1gQujXQQGOVNaiuwSN680Hq-FDVsCwvN-3AazykOBON0). First tab IS the main list
(row 1 = donations banner → `skiprows=1`; header has embedded newlines in a
few column names). Live has clean integer Year/Month/Day cols → used for
dates; it has NO `Surgery_count` column, so matching is on mlbamid + nearest
date. Snapshot = `TJS_Prediction/Raw_data/list of TJ.csv` (pulled ~2024-05).

## Cleaning (valid date AND mlbamid)
| source | total | dropped (no mlbamid) | dropped (no date) | clean |
|--------|------:|---------------------:|------------------:|------:|
| LIVE   | 2706  | 21                   | 0                 | 2685  |
| SNAP   | 2436  | 27                   | 0                 | 2409  |

Cleaned live saved: `tj_live_clean.csv` (all columns kept).
Sanity checks pass: snap latest date = 2024-05-01, snap 2024 = 26 dated rows.

## Dated-surgery counts per year (2016–2026)
| year | live | snap | live−snap |
|-----:|-----:|-----:|----------:|
| 2016 | 158  | 157  | 1  |
| 2017 | 160  | 159  | 1  |
| 2018 | 146  | 146  | 0  |
| 2019 | 139  | 136  | 3  |
| 2020 | 86   | 79   | 7  |
| 2021 | 172  | 163  | 9  |
| 2022 | 132  | 106  | 26 |
| 2023 | 130  | 85   | 45 |
| 2024 | 109  | 26   | 83 |
| 2025 | 71   | 0    | 71 |
| 2026 | 30   | 0    | 30 |

Gap ≈0 for 2016–2019, then grows monotonically toward the 2024-05 pull date —
the classic recording-lag signature (snapshot is a superset-deficient early
pull, not a date-disagreement issue).

## Surgeries in LIVE but MISSING from snapshot (unmatched live surgeries)
- Dated **2016–2023: 92** missing. Of these, 89 are players whose mlbamid is
  absent from the snapshot entirely; only 3 are second-surgery ("extra") cases.
- Dated **2022–2023: 71** missing (26 in 2022, 45 in 2023). **These are our
  test-window labels — each is a false negative in the current snapshot.**
- By year: 2016:1 2017:1 2018:0 2019:3 2020:7 2021:9 2022:26 2023:45.
- Examples (2016–2023, latest first): Gerelmi Maldonado 2023-12-01, Luke Kovach
  2023-12-01, Tyler Hardman 2023-10-10, Shohei Ohtani 2023-09-19, Alex
  McFarlane 2023-09-01, Reiss Knehr 2023-08-01, Jace Beck 2023-08-01, John
  Valle 2023-08-01, Penn Murfee 2023-07-01, Kenya Huggins 2023-07-01.

## Matched-pair date agreement
2408 matched pairs. 2407 have the exact same date; **only 1 pair differs
(by 6 days)**. The two sources essentially never disagree on a date — the
entire discrepancy is missing records, not shifted dates. (Only 1 snapshot
surgery is unmatched → snapshot is nearly a strict subset of live.)

## Recording lag (protopathic-relevant)
Live surgeries dated 2023-01 .. 2024-04: 183 total; **73 (40%) absent from the
2024-05 snapshot.** Misses span every month back to 2023-01 (15 missing that
month), so late additions reach back **≥16 months** before a pull; older years
(2016–2021) are ~complete, so the practical saturation tail is ~2 years.

## Recommended label-reliable end date
**2024-12-31** (primary). Rationale: live counts are saturated and consistent
with historical norm (~100–160/yr) through 2024 (109), then drop sharply at
2025 (71) — the signature of a year still being filled in. Because late
additions reach ~1.5–2 yr back, from a 2026-07 pull only dates ≲ end-2024 can
be trusted complete. **Conservative / safety-margin alternative: 2024-06-30**
(full 2-yr buffer). **2025–2026 must be excluded as incomplete** regardless.
Our test window (2022–2023) IS complete in the LIVE sheet — but NOT in the
snapshot (71 missing) → refresh labels from live before evaluating.
