# PROJECT_MEMORY — 공유 컨텍스트 (Claude Code · codex 공용)

이 파일은 두 에이전트가 동일한 전제에서 작업하기 위한 **단일 원본**이다.
진입점은 `.claude/CLAUDE.md`(Claude)와 `AGENTS.md`(codex)이며, 사용자가
"md 파일 업데이트"를 요청하면 **이 파일 + 두 진입점을 함께 갱신**한다.

## 1. 작업 원칙 (사용자 지시 이력 — 위반 금지)

- **권한 경계 (2026-07-13 최신 지시)**: `.md` 문서는 에이전트 판단으로
  갱신 가능하다. 그 외 기존 코드·데이터·설정 파일의 생성·수정, 다운로드,
  승인된 계획/실험 블록의 실행은 **사용자의 해당 세션 명시 지시 후에만**
  한다. 읽기 전용 점검과 파일을 남기지 않는 진단 재계산은 가능하다.
- 특히 학습(training)·GPU 작업은 방법 설명을 착수 명령으로 해석하지 않는다.
- **git commit/push/pull은 사용자가 직접.** 에이전트는 명시 요청 시에만
  git 상태를 건드리고, `.gitignore` 안전(아래 3절) 유지는 에이전트 책임.
  커밋 메시지에 Co-Authored-By 라인 금지. 위험 작업(merge/rebase/reset)은
  체크포인트 커밋 + 충돌 파일 명시 + 파괴적 플래그 금지.
- **라벨은 TJS-only 고정.** 팀원이 distance-based 단기 TJS 트랙을 수행
  중이므로 IL/일반 부상으로 라벨을 확장하지 않는다. arm-IL은 라벨이 아닌
  auxiliary feature 한정 — A-IL 블록에서 검정 완료(2026-07-13), blackout
  게이트 실패로 **canonical 금지, 공개 진단 후 triage 신호로만**.
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
  AND 마지막 등판이 t로부터 365일 이내(`days_since_last ≤ 365`). 라벨 =
  (t, t+H] 내 TJS, H ∈ {90, 150}.
- **fold**: train 2017–20 / valid 2021 / test 2022–24. primary 경계는
  **t+H ≤ 2024-12-31** (E0A 감사)이고, E0A가 함께 명시한 full 2-year
  safety 경계 `2024-06-30`도 민감도로 반드시 병기한다. 2025–26 라벨은
  미완결이다. 2025 성능은 이미 조회했으므로 untouched/prospective가 아니라
  향후 **label-refresh robustness set**이다. 진짜 전향 확인은 모델 동결 뒤
  처음으로 점수를 timestamp한 미조회 decision cohort부터 시작한다.
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
  동결 후 새 decision cohort의 라벨 성숙 시).
- **라벨 경계 민감도 (2026-07-13 실행 전 감사)**: E0A의 2-year safety
  경계 `t+H≤2024-06-30`에서는 H90 **0.660 [0.596, 0.726]** (사건 67),
  H150 **0.665 [0.603, 0.726]** (사건 56). 이는 primary를 철회하는 수치가
  아니라, 2024 강세 연도/경계 선택에 대한 필수 민감도다. 이때 RP-내 ROC는
  0.601/0.621, lift는 1.57×/1.43×(H150 P=0.072)로도 약해진다.
- **fit-side 불확실성 (A0-3)**: fit 투수 full-refit bootstrap CI
  [0.597, 0.702] / [0.600, 0.700]. 수술당 단일 landmark 민감도 −0.104
  (outcome-dependent stress test — 반복 landmark 구조가 성능의 실질
  일부, EPV 취약성). **"~0.60-0.70"은 공식 CI가 아니라 민감도들을 합친
  sensitivity envelope로 표기** (codex 3차).
- **KBO 관련 표현 (codex 3차 정정)**: novel 0.66(primary)-0.56(safety)
  은 **MLB novel-pitcher stress reference** — KBO 성능 예측으로 인용
  금지 (KBO 성능은 재적합 후 측정). A0-1 낙관 감사는 "3후보 평균
  −0.013, fold 변동 커서 확정 불가"로만 (수치적 "하한" 표현 금지).
- recall@50 = 21/75, 24/80; 무작위 대비 lift **1.76× / 1.55×** (MC
  2000회, P≤0.012 유의).
- subgroup(primary 경계): seen 투수 0.71 / **novel 투수 약 0.66**;
  SP-내 0.70 / RP-내 0.65 — 단 top-50 경보의 RP 사건 포착은 1건(경보
  배분 문제, "불펜 구조적 불가" 아님); 연도별 2022 0.68 / 2023 0.62–0.65 /
  2024 0.80(outlier). safety 경계의 novel ROC는 H90 0.572 / H150 0.557 —
  novel 대역 ~0.56–0.68은 **MLB novel-pitcher stress reference**로만
  인용 (KBO 성능 예측 아님 — codex 3차 정정; 0.66 단일값도 금지).
  [safety 경계 수치는 2026-07-13 fable 재계산으로 codex 보고와 일치 확인.]
- 보정: 원모델 calibration slope ~1.9 (top-decile ~2.2× 과소예측) — **절대 확률 인용
  전 validation 재보정 필수**. 순위 지표(ROC/recall)는 영향 없음.
- 문헌 비교는 스냅샷 추정량(~0.69)으로; **"문헌(0.61–0.67) 초과" 주장
  금지** (추정량 상이 + 비교 논문은 retrospective/CV).
- **인용 금지 목록**: Kang 0.93/재현 0.92(회고 아티팩트 — 재현 체크포인트
  로만), 0.816(저자 데이터+동결 라벨 한정), "무작위 대비 2–3배", 회고
  F1@0.5(threshold 미구현 임의 작동점), 2022-25 기반 0.689/0.693(라벨
  오염으로 철회됨).

## 4. 기각 이력 (재시도 전 근거 확인 — `results/phase3/`)

- **널/악화 확정**: Tier-2 트래킹 집계(달력/경기창/부분집합에서 일관된
  유의 증분 미검출; 모든 셀에서 +0.05를 배제한 것은 아님), tree/HistGBM
  (동급 또는 악화, 확장 그리드 무효), 시퀀스
  DL(사건 규모 부족), nested case-control(검정력 이득 없음), full GAM,
  역할 상호작용(M3)·within-role z(M4)·role-stratified stitch, age additive,
  Mayo식 추세(유의 악화), Mastroianni식 변동성(널), pitch-mix(유의 악화),
  rest 구조(dPR 악화), torque 가중(pc와 r=0.9998 무효), 경기 내 구속
  감쇠(널), M_bf 백필(rolling 비일관), **A1 RP 시간척도 feature 6종
  (pooled/interaction/분리 전부 널, H150 점추정 ≤0)**, **Cohen 2022
  release-drift×RP (사전 등록 1회 검정 널 — 회고→전향 이전 실패 추가
  사례)**, **A-IL 팔꿈치-IL 이력 (전체 +0.068/+0.046 EXCL0이지만
  blackout 60d에서 음수 — 공개 진단 메아리, "triage 신호"로만 분류,
  canonical 금지)**.
- **열린 후보**: prior_tjs(3개 test 구성 연속 방향 양 + recall 개선, CI 0
  포함 — 사건 축적/KBO에서 1순위 재검정), arm angle 드리프트(2020+만
  존재, train 커버리지 부족 — 시즌 축적 대기), A2 PITCHf/x train
  확장(보류). 새 결과는 동결 canonical의 "변형"으로만 병기 가능.

## 5. 현재 계획 (canonical: plan_progress.md 2026-07-13 절들)

- **[2026-07-13 실행 완료] A0(1-3) → A1 → A-IL → 동결** (사양 v2 =
  plan_progress.md "2026-07-13 (계속 3)"). 블록 결과 문서:
  `results/phase3/{A0_RESULTS, A1_BULLPEN, AIL_RESULTS, FROZEN_MODEL}.md`.
  - A0: 낙관 감사 −0.013(3후보 하한, 연도 노이즈 ±0.07 지배; 고정
    M_sa+hazard가 전 fold 선택보다 우수), 재보정 ≈ 항등(test slope
    ~1.9는 시대 shift — 배치 시점 재보정 필수 규칙 유지), 민감도(단일
    landmark −0.104, fit-side refit CI 하한 ~0.60).
  - A1: 모델 변형·Cohen 전부 널 (이번 6f/형태 한정 — 불펜 일반 종결
    아님). **quota q=20은 채택 철회 → 조건부 challenger** (primary
    Pareto 개선이나 **safety H150 게이트 실패 16→12**; codex 3차).
    canonical 경보 = q=0 top-50. q=5 사후 개선은 소급 채택 금지.
  - A-IL: blackout 게이트 실패 → **canonical 미채택** 불변. parser
    한계(elbow 소급 33건 등) 확인 — 공개일 정정 blackout도 ≈0.
    "재검정 필요한 triage 후보"로만 (검증된 triage 아님).
  - **동결 완료: `FROZEN_MODEL.md` + `frozen_model_state.json`
    (full-precision 정본, SHA-256 기록)이 단일 원본. 이후 MLB 모델 변경
    금지.** 2025 점수 아카이브 = frozen_scores_2025 (SHA-256 기록).
  - **2026년 4-7월 채점 = label-blind delayed shadow backfill** (진짜
    전향 아님 — 결정일이 채점일보다 과거; codex 3차 재분류). **첫 진짜
    전향 cohort = 2026-08-01을 당일 이전 채점·해시·저장** (상태 로드
    방식, append-only, 사용자 git commit/tag로 증빙).
- **남은 트랙**: KBO 이전 패키지(정의·계산법·검열/보정·role-aware 평가·
  경보 정책 이전, **계수·quota는 KBO 재적합/재선택**) + 제안서(사용자
  직접 작성, 재료: phase2_results 블록 6-9 + FROZEN_MODEL.md +
  PROJECT_RETROSPECTIVE.md) + 2026-08/09 채점 + 2025 robustness(~2027).
- **KBO 이전**: 이전 대상은 정의·feature 계산법·검열/보정 절차·role-aware
  평가·경보 정책 — **계수는 KBO 재적합** (MLB 계수 이식 금지). KBO
  부상자명단 제도는 내부 의무기록 접근을 요청할 근거이지만, 정확한 TJS/UCLR
  수술일·술식 DB가 존재한다는 뜻은 아니다. 먼저 label feasibility를 확인하고,
  primary는 TJS/UCLR 수술로 유지한다(비수술 UCL/arm-IL은 auxiliary/secondary).
  제안서는 사용자가 직접 작성(재료: phase2_results.md 블록 6–9).

## 6. 데이터·산출물 지도

- `data/raw/statcast_2016..2026.parquet` (2026 = 3/1-7/12 부분 스냅샷;
  gitignore)
- `data/prospective/`: cohort_v2/v3/v4, slim_games_v3/v4,
  game_features_v2/v3/v4, vdecay_games_v4, **gs_flags_v1** (A1 GS 플래그),
  tj_live_clean_20260707.csv (라벨 시트 스냅샷; md5 검증) — 전부 gitignore
- `data/ail/`: StatsAPI transactions 2016-2024 JSON + il_episodes.parquet
  (A-IL) — gitignore
- `results/phase3/`: R1_ROLLING/R2_AGE/R3_PREP/R4_KBO_LABELS,
  B_TIER_MODELS, B_PRIME_LEAD, NEXT_STEPS_DIAGNOSIS,
  **P_BLOCK_RESULTS(정정 헤더)**, **A0_RESULTS / A1_BULLPEN / AIL_RESULTS /
  FROZEN_MODEL(동결 단일 원본)**, frozen_scores_2025_ts20260713.csv
  (md5 a3304...) + 결과 CSV + `scripts/`(전 실험 재현 코드)
- canonical 문서 역할: `phase2_results.md`(결과 이력) /
  `plan_progress.md`(계획+로그) / P_BLOCK_RESULTS.md(수치 정정 이력) /
  **FROZEN_MODEL.md(동결 모델·인용 수치의 최종 단일 원본)**

## 7. 스타일 (요약)

- 한국어 기본. 옵션 매트릭스(A/B/C) + 명시적 권장. 가정 선(先)명시.
  결과는 정직하게(실패·스킵 포함) 보고. 이모지 금지, 결정 메모 스타일.
- 코드: 가정 표면화 → 최소 코드 → 외과적 수정 → 성공 기준 정의·검증
  (karpathy). Python: `from __future__ import annotations` + type hints,
  snake_case, 상대경로(`Path(__file__).parent`), 1–2줄 docstring.
- 환경: Windows 11 / PowerShell, uv 관리 `.venv/`(Python 3.11.11),
  torch는 cu128 인덱스.
