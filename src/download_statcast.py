"""Download raw Statcast pitch data season by season to data/raw/*.parquet.

Season range follows Kang (Mar 1 - Nov 30, 2016-2023) + 2024-2025 (P0-1/A1
test extensions, 2026-07-10) + 2026 partial (through 2026-07-12 snapshot,
approved 2026-07-13 for frozen-model prospective scoring only — 2026 labels
stay unused). The 2025 plate_x/z definition change does not affect the
workload/velocity/role features in use.
Upstream fetched the same ranges in parallel threads; we go sequentially —
pybaseball's cache makes reruns/resumes cheap and already-downloaded seasons
are skipped via the parquet file check.

Run: .venv\\Scripts\\python.exe src\\download_statcast.py
"""
from __future__ import annotations

import time
from pathlib import Path

from pybaseball import cache, statcast

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SEASONS = range(2016, 2027)
END_OVERRIDE = {2026: "2026-07-12"}  # in-season snapshot; frozen scoring needs data before 2026-07-01 only


def main() -> None:
    cache.enable()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in SEASONS:
        out = RAW_DIR / f"statcast_{year}.parquet"
        if out.exists():
            print(f"{year}: already downloaded, skip", flush=True)
            continue
        end_dt = END_OVERRIDE.get(year, f"{year}-11-30")
        print(f"{year}: fetching {year}-03-01 .. {end_dt}", flush=True)
        # Savant occasionally returns a malformed response for one day-query;
        # a retry (on top of pybaseball's day-level cache) gets past it.
        for attempt in range(3):
            try:
                df = statcast(start_dt=f"{year}-03-01", end_dt=end_dt, verbose=False)
                break
            except Exception as e:
                print(f"{year}: attempt {attempt + 1} failed ({type(e).__name__}: {e})", flush=True)
                if attempt == 2:
                    raise
                time.sleep(30)
        df.to_parquet(out)
        print(f"{year}: {len(df):,} pitches, {df.shape[1]} cols -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
