# M2 — immutable frozen-state scorer (2026-07-14)

## 결정

M2를 **PASS**로 판정한다. `frozen_model_state.json`의 full-precision
scaler·hazard LR·interval recalibration을 직접 로드하며, 미래 채점 경로는
학습 cohort·TJS 라벨·scikit-learn estimator와 `fit()`을 읽거나 호출하지
않는다. 기존 2025 archive와 2026 delayed-shadow archive는 이력 보존을 위해
수정하지 않았다.

## 구현

- `scripts/frozen_state.py`: SHA-256, 필수 schema, feature 순서, 벡터 길이,
  유한값, 양의 scaler scale을 검증하는 loader와 NumPy score equation.
- `scripts/prospective_scoring.py`: 명시적 decision date의 `game_date < t`
  cohort/feature 구축, actual first-pitch starter 판별, q=0 top-50,
  raw/recal P90·P150, feature·표준화 기여도 출력.
- `scripts/score_2026_prospective.py`: 위 모듈만 호출하는 미래 채점 CLI.
- CSV와 `.manifest.json`은 exclusive-create로 생성한다. 같은 경로 재실행은
  실패하며 state·input·score SHA-256와 decision/as-of metadata를 함께
  보존한다. 과거 decision date와 오래된 raw snapshot은 명시적 예외 없이는
  거부한다.

## 검증

- 환경: `scripts/verify_env.py` **9/9 PASS**. 최초 sandbox 실행은 사용자
  홈의 pybaseball cache 쓰기 권한 때문에 8/9였고, 승인된 정상 환경에서
  단일 날짜 fetch까지 재검증했다.
- `python -m unittest discover -s results/phase3/tests -v`: **9/9 PASS**.
- golden regression: 2026-04-01~07-01의 2,654개 delayed-shadow window를
  state-load equation으로 다시 계산해 기존 archive와 병합했다. P90/P150
  raw·recal 네 열은 전 행에서 절대오차 5e-8 이내, `rank_H150`은 전 행
  완전 일치했다.
- strict as-of: synthetic game at `game_date == t`가 career count, workload,
  velocity, GS share 모두에서 제외됨을 검증했다.
- q=0: 월 60명 fixture에서 정확히 rank 1–50만 alert됨을 검증했다.
- append-only: 같은 output/manifest 경로의 두 번째 쓰기가
  `FileExistsError`로 중단됨을 검증했다.
- 2026 regular-season first-pitch sanity: 모든 game이 정확히 두 starter를
  가졌다. 필수 순서 좌표 결측과 exact tie도 0건이며, 향후 입력에서는
  둘 중 하나라도 생기면 채점 전에 실패한다.

## 사용 경계

- 2026-08-01의 진짜 전향 snapshot은 8월 1일 당일 또는 이전에 최신
  `statcast_2026.parquet`(마지막 경기 < 8/1)로 별도 실행해야 한다. 이번
  M2 검증은 새 전향 점수를 미리 만들지 않았다.
- q20 열은 과거 challenger 기록으로만 보존한다. 새 snapshot의 canonical
  경보 열은 `alert_q0`이다.
- recalibrated probability도 window-level calibration 보장값이 아니므로
  dashboard에서 절대위험으로 해석하지 않는다.
