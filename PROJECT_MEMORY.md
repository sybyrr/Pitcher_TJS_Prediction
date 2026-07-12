# PROJECT_MEMORY — 공유 컨텍스트 (Claude Code · codex 공용)

이 파일은 두 에이전트가 동일한 전제에서 작업하기 위한 **단일 원본**이다.
진입점은 `.claude/CLAUDE.md`(Claude)와 `AGENTS.md`(codex)이며, 사용자가
"md 파일 업데이트"를 요청하면 **이 파일 + 두 진입점을 함께 갱신**한다.

## 1. 작업 원칙 (사용자 지시 이력 — 위반 금지)

- **학습(training)·GPU·새 실험 블록 착수는 사용자의 명시 지시 후에만.**
  방법을 설명하는 것과 착수 명령은 다르다. CPU 분석·검증은 진행 가능.
- **git commit/push/pull은 사용자가 직접.** 에이전트는 명시 요청 시에만
  git 상태를 건드리고, `.gitignore` 안전(아래 3절) 유지는 에이전트 책임.
  커밋 메시지에 Co-Authored-By 라인 금지. 위험 작업(merge/rebase/reset)은
  체크포인트 커밋 + 충돌 파일 명시 + 파괴적 플래그 금지.
- **라벨은 TJS-only 고정.** 팀원이 distance-based 단기 TJS 트랙을 수행
  중이므로 IL/일반 부상으로 라벨을 확장하지 않는다. arm-IL의 (라벨 아닌)
  auxiliary feature 사용은 **사용자 확인 전 금지** (미결정 상태).
- **외부 레퍼런스는 검증 후 채택**: 해당 연구의 데이터·라벨·평가 설계를
  우리 setup과 대조해 borrow/adapt/avoid를 명시한다. **수치 이식 금지**
  (retrospective/CV 수치를 우리 전향 수치와 직접 비교하지 않는다).
- **블록/세션 종료마다 문서 갱신** (세션 단절·compaction 대비). 정정은
  원문을 지우지 않고 정정 표기로 덧붙인다 (이력 보존).
- `TJS_Prediction/`(upstream 클론)은 **수정 금지**. 저자 `final_df.csv`는
  절대 GitHub에 올리지 않는다.
- Statcast 대량 다운로드는 `pybaseball.cache.enable()` 켜고 진행.

## 2. 동결 프로토콜 (재검토 없이 변경 금지)

- **코호트**: 결정일 = 매월 1일(4–9월); 적격 = 경력 ≥20경기(2016+ 기준)
  AND 마지막 등판 ≤365일 전. 라벨 = (t, t+H] 내 TJS, H ∈ {90, 150}.
- **fold**: train 2017–20 / valid 2021 / test 2022–24, 단
  **t+H ≤ 2024-12-31** (라벨 신뢰 종료 — E0A 감사; 늦은 등재 1.5–2년
  소급). **2025–26 라벨은 미완결** — 2025 창은 동결 모델의 향후 전향 확인
  세트 전용(라벨 성숙 ~2027 중반).
- **평가**: 투수-clustered bootstrap 1000회(seed 0)를 변형 간 공유 →
  paired ΔROC/ΔPR; event-level recall@k(결정일별 top-k, 사건 =
  (투수, 수술일)). 새 변형은 **anchor 게이트 재현 후** paired로만 판정.
  채택 기준 = CI 0 배제 또는 사전 등록된 방향 일관성(rolling-origin).
- **canonical 모델**: M_sa 9-feature {pc_chronic, pc_acute_dev,
  days_since_last, vel_trend, month, start_share, prior_pc_rate, ncg_log,
  vt_missing} + **discrete-time hazard supermodel**(plain LR, 30일 구간
  s=0..4, 누적곱으로 P(H) 산출; H90≤H150 정합 보장).

## 3. Canonical 수치 (2026-07-13 정정 기준) 및 인용 규칙

- **대표 수치**: H90 ROC **0.701 [0.643, 0.759]** (test 사건 75) / H150
  **0.696 [0.645, 0.746]** (사건 80). 반드시 "adaptive selection 이후
  **조건부 backtest**"로 표기 (untouched test 부재; 진짜 전향 확인은
  동결 후 2025–26 라벨 성숙 시).
- recall@50 = 21/75, 24/80; 무작위 대비 lift **1.76× / 1.55×** (MC
  2000회, P≤0.012 유의).
- subgroup: seen 투수 0.71 / **novel 투수 0.66 (KBO 이전 기대치 기준)**;
  SP-내 0.70 / RP-내 0.65 — 단 top-50 경보의 RP 사건 포착은 1건(경보
  배분 문제, "불펜 구조적 불가" 아님); 연도별 2022 0.68 / 2023 0.62–0.65 /
  2024 0.80(outlier).
- 보정: 재보정 slope ~1.9 (top-decile ~2.2× 과소예측) — **절대 확률 인용
  전 validation 재보정 필수**. 순위 지표(ROC/recall)는 영향 없음.
- 문헌 비교는 스냅샷 추정량(~0.69)으로; **"문헌(0.61–0.67) 초과" 주장
  금지** (추정량 상이 + 비교 논문은 retrospective/CV).
- **인용 금지 목록**: Kang 0.93/재현 0.92(회고 아티팩트 — 재현 체크포인트
  로만), 0.816(저자 데이터+동결 라벨 한정), "무작위 대비 2–3배", 회고
  F1@0.5(threshold 미구현 임의 작동점), 2022-25 기반 0.689/0.693(라벨
  오염으로 철회됨).

## 4. 기각 이력 (재시도 전 근거 확인 — `results/phase3/`)

- **널/악화 확정**: Tier-2 트래킹 집계(달력/경기창/부분집합 3중 널, +0.05
  초과 배제), tree/HistGBM(동급 또는 악화, 확장 그리드 무효), 시퀀스
  DL(사건 규모 부족), nested case-control(검정력 이득 없음), full GAM,
  역할 상호작용(M3)·within-role z(M4)·role-stratified stitch, age additive,
  Mayo식 추세(유의 악화), Mastroianni식 변동성(널), pitch-mix(유의 악화),
  rest 구조(dPR 악화), torque 가중(pc와 r=0.9998 무효), 경기 내 구속
  감쇠(널), M_bf 백필(rolling 비일관).
- **열린 후보**: prior_tjs(3개 test 구성 연속 방향 양 + recall 개선, CI 0
  포함 — 사건 축적/KBO에서 1순위 재검정), arm angle 드리프트(2020+만
  존재, train 커버리지 부족 — 시즌 축적 대기), A2 PITCHf/x train
  확장(보류), 불펜 mini-block(A1 — 승인 대기).

## 5. 현재 계획 (canonical: plan_progress.md 2026-07-13 절, 사양 포함)

- **승인된 실행 순서 [2026-07-13, compact 후 사용자 "시작" 지시 대기]**:
  **A0(1-3) → A1 → A-IL → 동결.** 상세 사양은 plan_progress.md
  2026-07-13 절들 — **사양대로 기계적 실행, 재설계 금지.**
  - A0(1-3): nested rolling-origin(선택 낙관 추정) → fit-set 5-fold OOF
    재보정 → hazard 사건-가중/s-범주형 민감도.
  - A1 불펜: GS 기반 역할 5분류 → RP 시간척도 feature(≤6) → Cohen 2022
    release-drift×RP 1회 → 경보 quota validation-Pareto (채택 = 총 예산
    고정, RP 포착↑ & 전체 recall 손실 ≤2건).
  - A-IL: MLB StatsAPI IL 트랜잭션 수집 → 팔꿈치-IL 이력 feature ≤3개
    additive 검정 (라벨은 TJS-only 불변 — feature 사용은 원 결정에서
    허용, MLB 실험은 2026-07-13 명시 승인) + 신규 정보·lead 분해 필수.
  - 동결: 전 블록 종료 후 FROZEN_MODEL.md (스펙+계수 고정; 2025-26 라벨
    성숙 ~2027 중반 시 1회 전향 확인).
- **KBO 이전**: 이전 대상은 정의·feature 계산법·검열/보정 절차·role-aware
  평가·경보 정책 — **계수는 KBO 재적합** (MLB 계수 이식 금지). 요청 =
  이미 존재하는 내부 라벨(부상자명단 제도, 2020~)에의 통제 접근. 제안서는
  사용자가 직접 작성(재료: phase2_results.md 블록 6–9).

## 6. 데이터·산출물 지도

- `data/raw/statcast_2016..2025.parquet` (gitignore)
- `data/prospective/`: cohort_v2/v3/v4, slim_games_v3/v4,
  game_features_v2/v3/v4, vdecay_games_v4, tj_live_clean_20260707.csv
  (라벨 시트 스냅샷; md5로 scratchpad본과 동일 검증됨) — 전부 gitignore
- `results/phase3/`: R1_ROLLING/R2_AGE/R3_PREP/R4_KBO_LABELS,
  B_TIER_MODELS, B_PRIME_LEAD, NEXT_STEPS_DIAGNOSIS,
  **P_BLOCK_RESULTS(최신 수치·정정 헤더)** + 결과 CSV + `scripts/`(전
  실험 재현 코드; v_codex.py = 정정 수치 재현)
- canonical 문서 역할: `phase2_results.md`(결과 이력) /
  `plan_progress.md`(계획+로그) / `results/phase3/P_BLOCK_RESULTS.md`(수치)

## 7. 스타일 (요약)

- 한국어 기본. 옵션 매트릭스(A/B/C) + 명시적 권장. 가정 선(先)명시.
  결과는 정직하게(실패·스킵 포함) 보고. 이모지 금지, 결정 메모 스타일.
- 코드: 가정 표면화 → 최소 코드 → 외과적 수정 → 성공 기준 정의·검증
  (karpathy). Python: `from __future__ import annotations` + type hints,
  snake_case, 상대경로(`Path(__file__).parent`), 1–2줄 docstring.
- 환경: Windows 11 / PowerShell, uv 관리 `.venv/`(Python 3.11.11),
  torch는 cu128 인덱스.
