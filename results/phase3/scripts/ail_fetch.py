"""A-IL step 1: fetch MLB StatsAPI transactions 2016-2024 (spec v2).

Monthly chunks from https://statsapi.mlb.com/api/v1/transactions
(sportId=1; codex audit showed minor-league rows still leak in, so we also
save the MLB team-id set from /api/v1/teams?sportId=1 for a toTeam filter
at parse time). Raw JSON saved to data/ail/transactions_YYYY.json
(data/ is gitignored). Information time downstream = transaction `date`,
never the retroactive effectiveDate.
"""
from __future__ import annotations
import calendar
import json
import time
from pathlib import Path
import requests

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
OUT = ROOT / "data" / "ail"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://statsapi.mlb.com/api/v1"
t0 = time.time()

teams = requests.get(f"{BASE}/teams", params={"sportId": 1, "season": 2024}, timeout=30).json()["teams"]
mlb_ids = sorted(t["id"] for t in teams)
(OUT / "mlb_team_ids.json").write_text(json.dumps(mlb_ids))
print(f"MLB team ids: {len(mlb_ids)}")

for year in range(2016, 2025):
    if (OUT / f"transactions_{year}.json").exists():
        print(f"{year}: exists, skip")
        continue
    rows = []
    for m in range(1, 13):
        start = f"{year}-{m:02d}-01"
        end = f"{year}-{m:02d}-{calendar.monthrange(year, m)[1]:02d}"
        for attempt in range(3):
            try:
                r = requests.get(f"{BASE}/transactions",
                                 params={"startDate": start, "endDate": end, "sportId": 1},
                                 timeout=60)
                r.raise_for_status()
                rows.extend(r.json().get("transactions", []))
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2.0 * (attempt + 1))
        time.sleep(0.4)
    (OUT / f"transactions_{year}.json").write_text(json.dumps(rows))
    print(f"[t={time.time()-t0:.0f}s] {year}: {len(rows):,} transactions")
print("done")
