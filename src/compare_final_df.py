"""Compare our rebuilt final_df against the authors' final_df.

Quantifies pipeline fidelity for Phase 0: cohort agreement, per-pitcher time
axis agreement (tests the [GAP-FILL] in extract.py), and per-column agreement
of the 102 shared diff features on rows matched by (pitcher, target,
new_before_tj).

Run: .venv\\Scripts\\python.exe src\\compare_final_df.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OURS_CSV = ROOT / "data" / "final_df.csv"
AUTHORS_CSV = ROOT / "TJS_Prediction" / "Raw_data" / "final_df.csv"


def cohort(df: pd.DataFrame) -> set[tuple[int, int]]:
    return set(map(tuple, df[['pitcher', 'target']].drop_duplicates().values))


def main() -> None:
    ours = pd.read_csv(OURS_CSV)
    theirs = pd.read_csv(AUTHORS_CSV)
    diff_cols = sorted(set(c for c in ours.columns if c.startswith('diff_'))
                       & set(c for c in theirs.columns if c.startswith('diff_')))
    print(f"ours: {ours.shape}, theirs: {theirs.shape}, shared diff cols: {len(diff_cols)}")

    # --- cohort agreement ---
    c_ours, c_theirs = cohort(ours), cohort(theirs)
    both = c_ours & c_theirs
    print(f"\ncohort: ours={len(c_ours)}, theirs={len(c_theirs)}, common={len(both)}")
    for label, only in (("ours-only", c_ours - c_theirs), ("theirs-only", c_theirs - c_ours)):
        names = {}
        src = ours if label == "ours-only" else theirs
        for pid, tgt in sorted(only):
            name = src.loc[(src['pitcher'] == pid) & (src['target'] == tgt), 'player_name'].iloc[0]
            names[f"{name}({pid})"] = tgt
        print(f"{label} ({len(only)}): {names}")

    inj_common = sum(1 for _, t in both if t == 1)
    print(f"common cohort by class: injured={inj_common}, normal={len(both) - inj_common}")

    # --- time-axis agreement per common sample (tests the [GAP-FILL]) ---
    key = ['pitcher', 'target']
    days_ours = ours.groupby(key)['new_before_tj'].agg(['min', 'max', 'count'])
    days_theirs = theirs.groupby(key)['new_before_tj'].agg(['min', 'max', 'count'])
    j = days_ours.join(days_theirs, lsuffix='_o', rsuffix='_t', how='inner')
    same_span = ((j['min_o'] == j['min_t']) & (j['max_o'] == j['max_t'])).mean()
    same_count = (j['count_o'] == j['count_t']).mean()
    print(f"\ntime axis (common samples): identical span {same_span:.1%}, "
          f"identical row count {same_count:.1%}")
    print(f"row-count difference (ours - theirs): "
          f"median {(j['count_o'] - j['count_t']).median():.0f}, "
          f"mean {(j['count_o'] - j['count_t']).mean():.2f}")

    # --- feature agreement on exactly matched rows ---
    merged = pd.merge(
        ours[key + ['new_before_tj'] + diff_cols],
        theirs[key + ['new_before_tj'] + diff_cols],
        on=key + ['new_before_tj'], suffixes=('_o', '_t'))
    print(f"\nrows matched on (pitcher, target, new_before_tj): {len(merged):,} "
          f"of ours {len(ours):,} / theirs {len(theirs):,}")

    corrs, mads = {}, {}
    for c in diff_cols:
        a, b = merged[f'{c}_o'], merged[f'{c}_t']
        ok = a.notna() & b.notna()
        if ok.sum() > 2:
            corrs[c] = np.corrcoef(a[ok], b[ok])[0, 1]
            mads[c] = (a[ok] - b[ok]).abs().mean()
    corr_s = pd.Series(corrs)
    print(f"per-column correlation over matched rows: median r={corr_s.median():.4f}, "
          f"min r={corr_s.min():.4f}, cols with r>0.99: {(corr_s > 0.99).mean():.1%}")
    print("worst 5 columns by r:")
    for c, r in corr_s.nsmallest(5).items():
        print(f"  {c}: r={r:.4f}, mean|diff|={mads[c]:.4f}")


if __name__ == "__main__":
    main()
