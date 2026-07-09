# R3 PREP — retrospective (Kang-style) dataset with LIVE labels

PREP ONLY. No GPU training was started. The main session launches v9o_snap / v9o_live itself.

## 1. extract.py — flag-gated CLI additions (default byte-identical)
`git diff src/extract.py` (24 insertions, 5 deletions):
- New `--tj-csv PATH`: overrides ONLY the TJ-list input. Wired via
  `merge_tj_labels(df, tj_csv=None)` (default None -> reads the same
  `TJS_Prediction/Raw_data/list of TJ.csv` snapshot). Verified against
  `merge_tj_labels`'s consumed columns: `mlbamid`, `TJ Surgery Date`, `Year of TJ`.
- New `--out PATH`: overrides ONLY the output final_df path. `cohort_meta.csv`
  path is deliberately unchanged (fixed `data/cohort_meta.csv`) per task spec.
- Parsing via a tiny `_arg_value(flag)` helper; the pre-existing
  `causal = '--causal' in sys.argv` idiom is untouched.
- Default path (no flags): `tj_csv=None`, `out_csv=data/final_df.csv` — identical
  to prior behavior. Validated by argv simulation.

## 2. Live TJ csv — data/tj_live_for_extract.csv
Built from scratchpad `tj_live_clean.csv` (2685 valid rows, all with mlbamid_i /
surg_date / Year). Extract-compatible 3-column schema:
- `mlbamid` = mlbamid_i (int64)
- `TJ Surgery Date` = surg_date reformatted to snapshot style YYYY-M-D (no zero pad)
- `Year of TJ` = Year (int64)
Rows: **2685**. Max 3 surgeries/pitcher (matches extract's k in {1,2,3}). 0 Year-vs-date mismatches.
Sample:
```
 mlbamid  TJ Surgery Date  Year of TJ
  808054         2026-7-1        2026
  700327         2026-7-1        2026
  702153         2026-7-1        2026
```

## 3. Extractions
- SNAPSHOT re-extraction (OURS_CSV): `data/final_df.csv` ALREADY EXISTED
  (samples=656, injured=111, normal=545). REUSED, not rebuilt.
- LIVE extraction:
  `python src/extract.py --tj-csv data/tj_live_for_extract.csv --out data/final_df_live.csv`
  runtime **2 min 42 s** (12:53:55 -> 12:56:37), rc=0.
  Output `data/final_df_live.csv`: 74,710 rows, **samples=652 (injured=124, normal=528)**, 102 diff feats.

### cohort_meta.csv side effect (IMPORTANT)
extract.py writes cohort_meta to the FIXED path `data/cohort_meta.csv` (independent
of `--out`). The live run overwrote the snapshot meta. Handled:
- `data/cohort_meta_snapshot.csv` — snapshot meta backup (657 rows, 111 inj / 546 norm)
- `data/cohort_meta_live.csv`     — live meta backup (653 rows, 124 inj / 529 norm)
- `data/cohort_meta.csv` — RESTORED to snapshot content (pre-task invariant; md5-verified).

## 4. Cohort deltas (ours-snapshot vs ours-live)
Sample = one (pitcher, target) group. Pitchers can be dual-role (both targets).

| cohort   | samples | injured | normal |
|----------|--------:|--------:|-------:|
| snapshot |     656 |     111 |    545 |
| live     |     652 |     124 |    528 |

Sample-level delta: **+13 injured samples added**, **0 injured dropped**;
0 normal added, 17 normal dropped. Net injured +13 (+11.7%), normal -17, total -4.

Per-pitcher status transitions (21 pitchers changed):
- normal -> injured: 9
- absent -> injured: 4   (=> 13 pitchers GAINED an injured sample)
- normal -> absent : 8   (new surgery label makes them fail injured-eligibility AND
                          drops them from the 4-consecutive-normal run)

The 13 gained-injured pitchers (all anchor in 2023 = last MLB game 2023), by the
surgery that flipped them:
- **11 dated 2024** (Musgrove, Javier, Manoah, Gray, Alzolay, Patiño, Raley,
  L.Ortiz, D.Smith'24, Urquidy'24, Garrett'24) — recovered because the live list
  extends to 2026; the snapshot was frozen earlier and lacked 2024 surgeries.
- **2 dated 2023** (Reiss Knehr 2023-08, Shohei Ohtani 2023-09).

### Cross-check vs the 71 missing 2022-23 surgeries
71 = live surgeries dated 2022-23 whose (mlbamid, year) is absent from the
snapshot's 2022-23 entries (Definitions A/B/D all = 71; saved missing_2223.csv, 70 unique pitchers).
- gained-injured pitchers in the 71-set: **2** (Knehr, Ohtani)
- 68 of the 70 unique missing pitchers never enter the Kang cohort at all
  (minor-leaguers / no eligible 2016-23 MLB Statcast seasons).
=> The 71 figure is about raw-registry completeness, NOT retrospective-cohort
impact. The cohort change is dominated by 2024 surgeries (11/13), out of scope
for the "2022-23" framing.

### Injured per anchor year (v9 temporal test = anchor_year >= 2022)
Injured by anchor year — snapshot vs live differ ONLY in 2023:
- snapshot: 2018:13 2019:14 2020:17 2021:19 2022:19 2023:29  -> **test(>=2022)=48**
- live:     2018:13 2019:14 2020:17 2021:19 2022:19 2023:**42** -> **test(>=2022)=61**  (+13)

Full temporal TEST set (anchor>=2022):
- snapshot: 191 total (48 injured / 143 normal); train+valid pool 465
- live:     194 total (61 injured / 133 normal); train+valid pool 458

=> The label correction raises injured test positives 48 -> 61 (+27%). The 13
added positives are true future TJ cases the snapshot had mislabeled as normal —
exactly the correction the live-label re-measure of v9 AUC 0.816 targets.

## 5. run_phase2.py VARIANTS (minimal, exact-dict per spec)
Added `LIVE_CSV = data/final_df_live.csv` and two entries:
```
'v9o_snap': dict(csv=OURS_CSV, data=dict(drop_time_channel=True), common_window=True, split='temporal'),
'v9o_live': dict(csv=LIVE_CSV, data=dict(drop_time_channel=True), common_window=True, split='temporal'),
```
Dry-check: `python -c "import src.run_phase2"` imports cleanly, NO training
started; both keys present; OURS_CSV / LIVE_CSV / META_CSV all exist.
Resume-safety: unchanged. Additions are new keys with new output filenames
(`results/phase2/v9o_snap.csv`, `v9o_live.csv`); per-seed append + done-set skip
logic in `run_variant` is untouched, so both variants are independently resumable.

## !! CRITICAL for the training session — META swap per variant
run_phase2 reads a SINGLE global `META_CSV = data/cohort_meta.csv` for the
temporal split's anchor years. The two variants need DIFFERENT metas:
- `v9o_snap` needs the SNAPSHOT meta (currently in place: cohort_meta.csv == cohort_meta_snapshot.csv).
- `v9o_live` needs the LIVE meta. Before running v9o_live:
  `cp data/cohort_meta_live.csv data/cohort_meta.csv`  (and restore snapshot after,
  or run all snapshot-meta variants first).
Running v9o_live against the snapshot meta (or v9o_snap against live meta) will
KeyError in `anchor_years_for` on samples whose (pitcher,target) differs between
cohorts. I did NOT modify run_phase2's anchor logic (mechanical-execution
guardrail; exact-dict spec). If hands-free execution is preferred, the minimal
enhancement is a per-variant `meta=` override defaulting to `META_CSV` (same
pattern as `csv=`) — flagged for the orchestrator, not applied.

## Anchors note
R3 produces only deterministic cohort COUNTS (no model fit). The snapshot
re-extraction reproduces the documented cohort (656 samples / 111 injured), the
relevant sanity anchor. The M0dp / M-role LR ROC/PR anchors pertain to the
PROSPECTIVE pipeline (cohort_v2 / slim_games), which R3 does not exercise.
