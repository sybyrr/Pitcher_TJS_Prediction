# FROZEN MODEL — MLB 전향 TJS 위험 지표 (동결 2026-07-13)

승인 블록(A0(1-3) → A1 → A-IL) 종료 후 동결. **이후 MLB 쪽 모델 변경
금지.** KBO 이전 시 정의·계산법·절차만 이전하고 **계수는 KBO 재적합**.
재현: `scripts/frozen_score.py` (계수·2025 점수 아카이브 생성 코드).

## 1. 코호트·라벨 (동결 프로토콜 그대로)

- 결정일 t = 매월 1일 (4-9월). 적격 = 경력 ≥20경기(2016+) AND 마지막
  등판 ≤365일. 라벨 = (t, t+H] 내 TJS (Roegele 시트), H ∈ {90, 150}.
- fit = 2017-2020 (train) + 2021 (valid) 창. 평가 신뢰 경계
  t+H ≤ 2024-12-31 (primary) / 2024-06-30 (safety).

## 2. Feature (M_sa, 9개; 전부 t 이전 정보만)

pc_chronic (90일 투구수/90) · pc_acute_dev (30일 rate − 90일 rate) ·
days_since_last · vel_trend (30일 가중 구속 − 직전 시즌 가중 구속;
결측 시 0 + vt_missing=1) · month · start_share (365일 내 ≥50구 경기
비율) · prior_pc_rate (직전 시즌 총투구/183) · ncg_log (log1p 경력
경기수) · vt_missing.

## 3. 모델 형태·계수

Person-period 확장 (30일 구간 s=0..4, 수술 시 검열) + plain LR
(class weight 없음), P(H) = 1 − ∏(1 − h_s) (H90: s<3, H150: s<5) —
P90 ≤ P150 정합 보장. 표준화 [M_sa, s] 위 계수 (frozen_score.py 출력):

| 항 | 계수 | scaler mean / scale |
|---|---|---|
| intercept | −5.900480 | — |
| pc_chronic | −0.214385 | 3.391374 / 4.456661 |
| pc_acute_dev | −0.083979 | 0.803412 / 2.776353 |
| days_since_last | −0.153668 | 95.919829 / 119.257344 |
| vel_trend | −0.011728 | −0.226795 / 1.194625 |
| month | −0.233497 | 6.570512 / 1.704943 |
| start_share | +0.110833 | 0.313505 / 0.419603 |
| prior_pc_rate | +0.133170 | 5.690833 / 4.713477 |
| ncg_log | −0.085616 | 4.296018 / 0.678813 |
| vt_missing | −0.010479 | 0.434288 / 0.495663 |
| s | −0.196999 | 1.993537 / 1.414392 |

재보정 (fit-set 투수-grouped OOF, 구간 수준): **a = −0.164030,
b = 0.966471**. **주의(codex 3차 정정 반영): 구간 수준 항등이 window
수준 P(H) 보정을 의미하지 않으며(fit-set window slope 0.24/0.55), test
slope ~1.9의 원인도 단정 불가 — 절대 확률은 배치 시점 최신 라벨 재보정
전 인용 금지. 순위 사용이 primary.**
**정본 상태**: 위 표의 반올림 값이 아니라 **`frozen_model_state.json`**
(full-precision scaler·계수·재보정; SHA-256
e14ba800227a5b65a12ca55114e106e20a4636857ef947d5997b9e496e02fac8)이
동결 모델의 단일 정본이다. 향후 채점은 재적합이 아니라 이 상태를
로드해 수행한다.

## 4. 경보 정책 [정정 2026-07-13, codex 3차 감사]

- **canonical 경보 = q=0**: 월별 top-50, 순위 점수 = P150 (H90/H150
  확률은 공동 출력). recall 등 canonical 성능 보고도 q=0 기준.
- **q=20 RP 예약은 조건부 challenger로 강등** (채택 철회): primary
  경계에서는 Pareto 개선(RP 0→5, 총 손실 2)이지만 **safety 경계
  H150에서 게이트 실패(16→12, 손실 4 > 허용 2)**. RP = trailing 365d
  GS share ≤0.2 (gs_flags_v1). 기존 점수 파일의 alert_q20 열은 당시
  challenger 기록으로 보존.
- 사후 관찰된 q=5(전 셀에서 총 포착 +1, RP +1)는 **소급 채택 금지** —
  KBO/전향 확인에서 재검정할 후보로만 기록.
- **KBO 이전 시 quota는 처음부터 재선택.**

## 5. 성능 인용 (canonical, 변경 금지)

- primary (t+H≤2024-12-31): **H90 ROC 0.701 [0.643, 0.759] (사건 75) /
  H150 0.696 [0.645, 0.746] (사건 80)** — "adaptive selection 이후
  조건부 backtest" 표기 필수.
- 항상 병기: safety 경계(≤2024-06-30) **0.660 / 0.665**; fit-side
  full-refit bootstrap CI [0.597, 0.702] / [0.600, 0.700] (A0-3);
  스냅샷 추정량 ~0.69. **"~0.60-0.70"은 공식 신뢰구간이 아니라 여러
  조건부 민감도를 합친 sensitivity envelope로 표기한다.**
- novel 투수 0.66(primary)–0.56(safety): **MLB novel-pitcher stress
  reference**다 — KBO 성능 예측이 아니며, KBO 성능은 재적합 후에만
  측정 가능 (기대 관리용 참조 대역으로만 인용).
- 선택 낙관 감사(A0-1): 사전 고정 3후보 감사에서 평균 −0.013, 단
  fold 변동(±0.07)이 커서 낙관 유무 확정 불가; 전체 탐색 낙관은
  미측정 (수치적 "하한" 표현 금지 — codex 3차 정정).
- recall@50 = 21/75, 24/80 (q=0); lift 1.76×/1.55× (primary 경계).

## 6. 전향 확인 프로토콜

- **2025 창 = label-refresh robustness set** (이미 조회되어 untouched
  아님). 동결 모델 점수를 라벨 성숙 전에 아카이브:
  `frozen_scores_2025_ts20260713.csv` (3,994 창 × 6 결정일, P90/P150
  raw·recal, rp_flag, alert_q20; **md5
  a330434447f1da68a9d83a039beed79e**). 라벨 성숙(~2027 중반) 시 이
  파일로만 평가 (재채점 금지).
- **2026년 4-7월 채점 [정정 2026-07-13, codex 3차]**: 산술은 검증됐지만
  (2,654 창, md5 1f7648ef..., SHA-256 ab7e8b87...) **진짜 전향 평가가
  아니다** — 결정일(4/1-7/1)이 동결·채점일(7/13)보다 앞서고 2026 수술
  일부가 이미 라벨 시트 스냅샷에 존재. 정확한 분류 =
  **label-blind delayed shadow backfill** (라벨 미사용 사후 채점).
  채점은 동일 fit set 재적합 + 계수 재현 assert 방식이었음
  (`score_2026_prospective.py`) — "재적합 없음" 표현 정정; 향후 채점은
  `frozen_model_state.json` 로드로 수행.
- **첫 진짜 전향 cohort = 2026-08-01**: 8월 1일 당일 또는 이전에
  채점·해시(SHA-256)·저장해야 성립. 이후 매 결정일 append-only로
  누적하고, timestamp 증빙은 사용자의 git commit/tag로 고정할 것
  (에이전트는 커밋하지 않음 — 사용자 작업).
- 라벨 평가는 성숙(~2028) 후 보존된 점수 파일로만 (재채점 금지).

## 7. 동결 이후 허용되는 것

- KBO 이전 작업 (재적합·재보정·quota 재선택 — MLB 계수 이식 금지).
- 열린 후보의 "별도 변형" 실험 (prior_tjs, arm angle 등) — 단 canonical
  수치·이 문서는 불변, 새 결과는 변형으로만 병기.
- triage 계층 (A-IL 분류) — canonical 밖의 별도 산출물.
