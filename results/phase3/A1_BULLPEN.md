# A1 블록 — 불펜 mini-block (2026-07-13, 사양 v2)

> **정정 (2026-07-13, codex 3차 감사 수용 — fable 재계산 검증):**
> 아래 quota "채택"은 **철회, q=20은 조건부 challenger로 강등.** 인용
> 규칙상 필수 병기인 safety 경계(t+H≤2024-06-30)에서 게이트를 재검하면
> H90은 통과(15→13, RP 0→4)하지만 **H150은 실패(16→12, 총 손실 4 >
> 허용 2, RP 0→2)** — fable 재계산이 codex 표와 완전 일치. 정확한 지위:
> **canonical 경보 = q=0(순수 top-50)**; q=20은 "primary 경계 통과,
> safety H150 실패"의 RP-coverage challenger. 사후 관찰된 q=5의 전 셀
> 개선(포착 +1, RP +1)은 **소급 채택 금지** — KBO/전향 확인 시 재검정
> 후보로만 기록. 기존 2025/2026 점수 파일의 alert_q20 열은 당시 생성된
> challenger 기록으로 보존(재생성 금지). 또한 "feature/모델 경로 종료"는
> "이번 6-feature 셋과 3개 모델 형태가 미통과"로 한정해 읽을 것 —
> 불펜 모델링 일반의 종결 증명이 아니다. 본문은 이력 보존을 위해 유지.

사양: `plan_progress.md` "2026-07-13 (계속 3)". H150이 채택 판정 primary
(시스템 출력은 H90/H150 공동 유지). mature test = t+H≤2024-12-31, paired
공유 bootstrap(seed 0), 게이트 M_sa 이진 H90 = anchor 0.69203 재현 확인.
코드 `scripts/{a1_gs_build,a1_bullpen}.py`, 수치 `a1_bullpen.csv`, 데이터
`data/prospective/gs_flags_v1.parquet`.

## 역할 재정의 (GS 기반)

- GS = raw statcast에서 (game_pk, inning_topbot)별 첫 타석 투수. 연간
  4,857-4,864 (경기당 정확히 2, 2020년 1,796) — sanity 통과.
- **모델링 3분류** (trailing 365d GS share): SP ≥0.5 / RP ≤0.2 / swing.
  test 22-24 분포: RP 7,888 / SP 3,646 / swing 551.
- 5분류는 서술용: RP는 long(등판당 ≥25구) 1,226 / short 6,662로 갈리고,
  opener·bulk(GS share>0.2 & GS당 <40구)는 **0건** — 이 코호트(경력
  ≥20경기)에서 opener 전담은 관측되지 않음. 모델에는 미사용.

## 모델 변형 — 전부 미채택 (H150 primary 기준 널)

RP 시간척도 feature 6개 고정: pitches_7d/14d, appearances_14d,
b2b_30d, 3-in-4_30d, last_outing_spike(직전 등판 − 90d 중앙값).

| 변형 (M_sa 대비 paired) | H90 dROC [CI] | H150 dROC [CI] | RP-내 ROC (H150) |
|---|---|---|---|
| baseline M_sa | — (ROC 0.692) | — (ROC 0.692) | 0.658 |
| pooled +6f | +0.009 [−0.007,+0.026] | −0.001 [−0.013,+0.010] | 0.653 |
| role-interaction (+rp+6+6×rp) | −0.004 [−0.038,+0.031] | −0.010 [−0.039,+0.016] | 0.614 |
| SP/RP 분리 (ridge C=0.1) | +0.007 [−0.037,+0.048] | +0.000 [−0.035,+0.034] | 0.647 |
| Cohen relx-drift ×RP (1회 검정) | −0.010 [−0.043,+0.019] | −0.008 [−0.036,+0.021] | 0.627 |

- 전 변형 CI 0 포함, H150 점추정 ≤0 → **feature/모델 경로 종료** (B 블록
  트래킹 널·P1 널과 정합). Cohen 검정은 사전 등록대로 널 → 즉시 종료
  (회고 case-control 신호의 전향 이전 실패 사례에 추가).
- 진단 유지: baseline RP-내 ROC 0.65-0.66 — RP 판별력은 이미 있고,
  문제는 경보 배분이다.

## 경보 quota — 채택 (정책 변경, 모델 불변)

- 안정 영역 규칙: pre-test rolling fold {2019, 2021}(2020 단축 제외)에서
  grid {0,5,10,15,20} 전부 총 사건 손실 ≤1 → **q\*=20** (최대).
- mature test 1회 적용 (canonical hazard 점수, 예산 50 고정):

| H | q=0 총/RP | q=20 총/RP | 게이트 (RP↑ & 총 손실 ≤2) |
|---|---|---|---|
| 90 | 21/75, RP 0/37 | 19, RP **5** | 통과 — 채택 |
| 150 | 24/80, RP 0/40 | 22, RP **5** | 통과 — 채택 |

- **해석: 같은 50건 예산에서 RP 예약 20석이 RP 사건 포착을 0→5로 올리고
  총 recall은 2건 감소** — 모델 ROC 개선이 아니라 배분 정책의 Pareto
  선택이며, 그렇게만 인용한다. 운영 정책 = "월별 top-50 중 RP 상위 20석
  예약"으로 동결 대상에 포함.
- 주의: fold 사건 수(12/17)가 작아 q\* 자체의 정밀도는 낮다. 게이트가
  test에서 통과했으므로 채택하되, KBO 이전 시 quota는 반드시 재선택.
