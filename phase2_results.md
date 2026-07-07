# Phase 2 / 2.5 결과 (canonical)

작성 기준일: 2026-07-07. 실행: 2026-07-07 야간 자율 세션 (B안 전체 스코프).
근거 데이터: `results/phase2/*.csv` (변형별 10 seeds, seed 100..1000 통일),
`results/phase2/prospective.csv`. 발견의 출처는 `phase2_findings.md`,
실행 과정 로그는 `phase2_worklog.md`. 모든 변형은 ViT(분류)·1D-CNN(회귀),
G1 재현 baseline과 동일 하이퍼파라미터. p값은 v1(분류)/r0std(회귀) 대비
seed-paired Wilcoxon.

## 헤드라인 결론

1. **회귀(수술까지 일수)의 신투수 일반화는 0이다.** 투수 단위 split만으로
   R² +0.79 → **−0.15** (전 seed 음수, p=0.002). causal feature로도 복구 불가
   (−0.08). 논문의 R² 0.79는 전부 "이미 본 투수의 시점 보간"이었다.
2. **분류 성능의 상당 부분은 fill-artifact 지름길이다.** 이력 길이 스칼라
   하나로 AUC 0.894. 지름길을 구조적으로 제거(공통 관측 창)하면 AUC
   0.94→0.82 (p=0.002). **다만 아티팩트 제거 + 미래 시즌 테스트에서도
   AUC 0.816의 실질 신호가 남는다** (v9 — 이 프로젝트의 "정직한 회고 수치").
3. **현 설계의 전향(prospective) 예측은 무작위 수준이다.** rolling
   as-of-date(730일 창, 150일 horizon)에서 ViT·LR 모두 PR-AUC ≈ base rate
   (0.020 vs 0.019), ROC-AUC 0.53–0.56. 회고 신호가 "지금 시점 데이터만으로
   미래 예측"으로는 이전되지 않았다 — G3(KBO 부상 지표)의 전제가 현
   설계로는 불충족.

## 분류 변형 (ViT, 10 seeds, f1_injured / ROC-AUC / PR-AUC)

| 변형 | 무엇을 바꿨나 | f1 | AUC | PR-AUC | p(vs v1) |
|---|---|---|---|---|---|
| v0_g1 | G1 재현 baseline | 0.714 | 0.920 | 0.797 | — |
| v1_bugfix | deepcopy + ViT shuffle | 0.717 | 0.920 | 0.804 | — |
| v2_f102 | +시간 채널 제거 (F=102) | 0.718 | 0.941 | 0.819 | 0.027 |
| **v3_window** | +공통 관측 창 100–700 (지름길 제거) | 0.491 | **0.821** | 0.544 | **0.002** |
| v4_lininterp | +진짜 선형보간 | 0.677 | 0.928 | 0.801 | 0.32 |
| v5_trainstat | +4.7σ train-only fit | 0.701 | 0.922 | 0.789 | 0.77 |
| v6_grouped | +실투수 단위 split (이중소속 8명 교정) | 0.656 | 0.911 | 0.764 | 0.28 |
| v6r (우리 데이터) | v6를 재추출본으로 | 0.669 | 0.911 | 0.751 | 0.70 |
| v7_temporal | 고정 미래 테스트 (anchor≥2022, 부상 43) | 0.649 | 0.897 | 0.769 | 0.064 |
| v8_causal | causal diff (우리 데이터, grouped) | 0.665 | 0.924 | 0.789 | 0.85 |
| **v9** | **지름길 제거 × 미래 테스트** | 0.549 | **0.816** | 0.608 | **0.002** |
| (참고) span 스칼라 단독 | max_real_bin 하나 | — | 0.894 | — | — |

해석: 학습 버그(v1)·시간 채널(v2)·보간 방식(v4)·통계 누수(v5)·분할
방식(v6)·causal feature(v8)는 전부 **비유의 또는 소폭**. 유의한 하락은
오직 **지름길 제거(v3, v9)**뿐 — 논문 AUC 0.93 중 ~0.11은 코호트 구성이
만든 이력-길이 아티팩트, 나머지 0.82가 투구 역학 신호로 추정된다.
v8이 성능을 유지하므로 **배치 가능한(causal) feature 정의는 공짜**다.

## 회귀 변형 (1D-CNN, 10 seeds)

| 변형 | R² | RMSE 0–100 | novel 투수/test 투수 | p(vs row) |
|---|---|---|---|---|
| r0std_row (논문 프로토콜) | **+0.790 ± 0.016** | 93.5 | 0.4 / 96 | — |
| r1_grouped (투수 단위) | **−0.153 ± 0.139** | 207.8 | 20 / 20 | 0.002 |
| r1o (우리 데이터) | −0.081 ± 0.106 | 215.9 | 22 / 22 | 0.002 |
| r2_causal | −0.079 ± 0.104 | 221.4 | 23 / 23 | 0.002 |

"수술까지 남은 일수"를 개별 경기 스냅샷에서 예측하는 태스크는 신투수
대상 신호가 없다. 이 회귀 축은 KBO 제안에서 **제외**해야 한다.

split 공정성 논거 (2026-07-07 논의): "이미 부상 확정된 투수 한정 태스크"라는
프레임에서도 grouped split이 유일한 배치 정합 평가다. 라벨(수술까지 일수)은
수술일에서 역산되므로 수술일이 알려진 투수에게만 존재하고, row split은 같은
투수의 다른 경기 라벨을 train에 노출해 test 라벨이 날짜 산수로 결정된다
(투수별 diff 패턴이 식별 지문 역할). 배치에서 회귀를 적용할 대상은 정의상
수술일 미상 → 라벨 관점에서 전원 novel 투수. 단 음수 R²의 해석 한계 둘:
수술 시점은 행정적 결정이라 라벨 노이즈가 크고, 현 모델은 단일 경기
스냅샷만 사용(개인 내 추세 미사용) — 추세 기반 설계는 Phase 3 A안 후보.

## Phase 2.5 — 전향적 재정의 (rolling as-of-date)

설계: 결정시점 = 2017–2023 시즌 중 매월 1일(4–9월), trailing 730일
(146×102, causal diff, 시점 t 이전 정보만), 라벨 = t 후 150일 내 TJS.
14,123 윈도우 / 양성 217 (1.54%). 시간 분할: train 2017–20 (7,404/102),
valid 2021 (2,179/31), test 2022–23 (4,540/84, base 1.85%).

| 모델 | PR-AUC | ROC-AUC | p@50 | 비고 |
|---|---|---|---|---|
| LR (요약 feature) | 0.024 | 0.564 | 0.00 | base rate 0.019 |
| ViT (3 seeds) | 0.020–0.021 | 0.53–0.55 | 0.02–0.04 | 무작위 수준 |

**전향 신호 검출 실패.** 유보: 이것은 하나의 설계점이다 — (a) 양성이
train 102개뿐, (b) 730일/150일 창·horizon은 첫 선택, (c) 라벨이 TJS만
(IL/부상 전반 아님), (d) 수술 직전 투수는 등판 중단으로 활동 필터에서
빠져 "가장 신호가 강한 시점"이 표본에서 제외되는 구조적 문제. 단, 회귀
붕괴(r1)와 방향이 일치해 "개인 내 미세 드리프트로 시점을 맞추는" 종류의
신호는 약하다고 보는 게 현재 증거에 부합한다.

## Phase 2.6 — 전향 재설계 조사·진단 (2026-07-07, ultracode 워크플로우)

조사 5각도 + 레퍼런스 검증 5건 + CPU 진단 실험 2건. 사용자 확정 제약:
**라벨 TJS 고정**(팀원 distance-based 단기 TJS 트랙과 충돌 방지 — IL/부상
전반으로 확장 금지), **전향 프레임 유지**, GPU 학습 없음. 스크립트·산출물:
scratchpad `X1_SUMMARY.md`, `X2_results.md`, `x1_*.py`, `x2_features_lr.py`,
`slim_games.parquet`(154,207 경기 캐시). 워크플로우 저널:
`subagents/workflows/wf_054eb7d3-d09/journal.jsonl`.

**결론: 무신호(2.5)의 원인은 고칠 수 있는 설계 결함 2 + 진짜 상한선 1.**

실험 X1 (라벨 공급 × informative censoring, 우리 데이터 정확 수치):
- horizon별 양성 윈도우 (train≤2020 / test22-23): 90d 70/59, **150d(현) 102/84**,
  365d 231/201, 548d 357/241. 150→365d에서 약 2.4배, test base 1.85%→4.43%.
- censoring: has-history 수술 69건 중 마지막 등판→수술 gap 중앙값 **54일**,
  **>30일 65.2%**(현 prev-30d 규칙에서 부적격), >60일 47.8%.
- coverage: 2022-23 has-history 수술 81건 중 현 규칙으로 catchable **39건(48.1%)**.
  누락 42건 분해: off-season 8, ≥20경기 미달 20, **informatively censored 14**.
- **핵심: recency를 30→60→90일로 넓혀도 회수 0건**(수술 전 몇 달씩 dark).
  recency gate를 **제거**하고 ≥20경기 활동 바닥만 두면 14건 전부 회수 →
  coverage 65.4%. horizon 365d만으로도 48.1%→66.7%. 상한 ~69%(나머지는 <20경기).

실험 X2 (workload/velocity 신호 탐침, CPU LR, 우리 데이터):
- 7-feature(acwr·days_since_last·pc_30/90·ng·vel_trend) test: **ROC 0.638**
  (CI 0.578-0.696, base 0.564 대비 실질 상승, valid 0.618 일치), PR-AUC 0.0294
  (CI가 baseline 0.024와 겹침 → 정밀도 이득은 노이즈), **prec@50 0.02**(=base rate,
  상위 랭킹 실효 향상 없음). days_since_last 단독 ROC 0.584.
- **구조적 확인: days_since_last가 현 eligibility로 30에서 잘림** — 정작 가장
  강할 비활동 신호가 절단됨(X1과 정합).

검증 통과 레퍼런스 (전부 `applicable_with_changes` — directly 없음):
| 레퍼런스 | borrow | avoid |
|---|---|---|
| van Houwelingen 2007 (landmarking) | 우리 월별 설계 = 교과서 landmarking; pooled-logistic 이산시간 hazard 형식 | prev-30d gate는 표준 left-truncation 아님(at-risk 삭제); Cox/Breslow 불필요 |
| Mayo 2021, OJSM 9(6) — UCLR 223명 case series, 수술 전 **최종 15경기 Mann-Kendall** | trend feature 스펙: 4FB velo τb −.657 / SL velo −.524 / 2FB velo −.429 / 4FB spin −.581 / CT spin +.410 / CU 사용률 +.486 (전부 p<.05) | 무대조군(피로 교란) → 효과크기·성능 이식 금지, 방향·창 설계만 차용 |
| Mastroianni 2026, AJSM 54(3):694-704 — case-control **78/156, workload 매칭** (사용자 제공, 2026-07-07 원문 검증) | variability feature 스펙: 등판 내 velo SD p=.012·수평 release SD p=.005(기저), release drift(계절 내·오프시즌), extension 변동성, **최종 5등판 velo 점진 하강 p=.019**; "자기 이력 대비 벤치마킹" 프레임. workload 매칭 → 부하와 **독립인** 신호 | 매칭 설계 → base rate/성능 기대 이식 금지. 최종 5등판 앵커가 IL 등재일 → IL 날짜는 앵커/feature로만, 라벨은 TJS 유지 |
| Dillon 2025, OJSM 13(12) — **급성** 파열 7/14, PCA+Mahalanobis | 급성 아형은 다등판 점진 하강 **없음** — 부상 경기 내 급변(velo −2.1SD, arm angle −1.5SD)만 → 월별 결정 시점에서 원리상 못 잡는 잔차 존재(정직한 상한 논거) | 수치·회고 설계 |
| ssharpe42/mlb-injury (Hawkes) | IL/transaction 데이터 배관을 **feature**로(수술일 de-lag) | body-part 라벨 채택(라벨 확장 금지); Hawkes 본체(희귀·비반복 사건에 부적합) |
| Riley 2020 (표본수) | 84 events는 안정 추정 바닥(~116-200) 미달 → 넓은 CI 명시(G2 강화) | "underpowered라서 숨은 신호 있다" 주장 금지 |

정정 (2026-07-07 재검증): 종전 표의 "Dillon et al." 행은 같은 그룹의 두 논문을
혼동한 것 — velo SD·release SD 변동성 소견은 **Mastroianni 2026**(78/156)의
것이고, Dillon 2025(급성 7/14)의 소견은 Mahalanobis 누적 이탈 + 경기 내 급변.
두 논문의 대비(만성: 점진 하강 있음 p=.019 vs 급성: 없음)가 아형별 신호
분리 가설을 직접 지지.

문헌 상한선(검증됨): **진짜 전향 투수-부상 모델은 AUC 0.61-0.67이 천장**
(Karnuta elbow 0.61, PMC combined 0.66, Rendar rookie-UCL 0.674 — 모두
{TJ-only, temporal, forward} 중 최소 하나 완화). 회고/case-control은 팽창
(Whiteside 72-75%, Kang 0.93). 즉 우리 0.53-0.56은 더 엄격한 설정의 정직한
연장. → **modest lift가 성공이며, "최초의 엄격 전향·TJS-only 벤치마크"**
자체가 산출물.

**권장 재설계 (우선순위·전향 유지·CPU):**
1. **eligibility gate 교체(최고 가치)**: `prev-30d 1경기` → `≥20 career games`
   활동 바닥만; days-since-last·단기 활동을 시변 feature로(절단 해제).
   **전제 리스크 = protopathic 누수**(수술 예정으로 인한 shutdown을 읽으면
   circular): 착수 전 수술일 lag 감사(ssharpe42 IL-date 소스) + feature창과 t
   사이 blackout/지연 horizon + ±N주 수술일 민감도 검사가 조건.
2. **horizon 365d(또는 90/150/365 다중) + recency 가중 feature**: 양성 ~2.4배·
   off-season 회수·추정 안정. 단 급성 신호는 근접(2-4주) → 730일을 146등분
   평활 대신 최근 bin 보존. 548d는 라벨 노이즈로 비권장.
3. **문헌 지지 feature 추가** — 3족: (a) workload/활동(raw acute/chronic
   투구수·BF·이닝·휴식·등판빈도·days-since-last; X2 실측 ROC +0.07)
   (b) 단기 **추세** [Mayo 2021 스펙]: 구종별 velo/spin Mann-Kendall τ 또는
   Sen slope, 최근 5·15경기 창 + 구종 사용률 drift (c) **변동성/반복성**
   [Mastroianni 2026 스펙]: 등판 내 velo SD·수평 release SD(자기 이력 대비),
   release drift, extension 변동성, 최근 5등판 velo 기울기 + 절대 velo level,
   선발/불펜 역할. (b)(c)는 만성/마모형 담당 — 급성 아형(Dillon 2025)은
   원리상 잔차로 남음을 기대치에 반영.
4. **추정·평가**: 이산시간 hazard/pooled person-period logistic(현 LR이 이미 이
   형식), 투수 클러스터 SE, 시간분할(중첩 윈도우 경계 넘김 금지), 선택적
   competing-risk(TJS/이탈/활동)→누적발생 위험 순위. 지표는 PR-AUC + 고정
   경보예산 하 precision@k/민감도 + calibration + decision-curve(ROC·accuracy
   헤드라인 은퇴), 투수-grouped bootstrap CI, rolling-origin으로 event 수 보강.
   null 지속 시 "이 통계력에서 실효 신호 없음"이 정직한 G2 산출물.

**확정 대기 결정(사용자)**: ① eligibility 교체 착수(권장, 단 수술일 감사가
선행) ② horizon 150→365 또는 다중(권장 365) ③ 추정 machinery 범위(권장:
pooled-logistic hazard부터, competing-risk는 이후). 확정 후 Phase 2.6 실험 착수.

## G2/G3 함의 및 다음 결정 포인트

- **G2 (정직한 수치) 확정**: 회고 판별 AUC 0.82 (아티팩트 제거, 미래
  시즌), 회귀 R² 없음(음수), 전향 예측 무작위 수준(현 설계).
- **G3 (KBO 부상 지표)**: 현 증거로는 "경기 데이터만으로 100일 전 예측"
  류의 제품 주장을 **뒷받침할 수 없다**. 정직한 KBO 스토리 후보:
  (i) 회고적 위험 판별(AUC 0.82)을 "위험 프로파일링/사후 분석" 도구로
  프레이밍, (ii) 전향 설계의 반복 (IL 전체 라벨, horizon·창 탐색,
  생존분석/랜드마킹, 활동 중단 신호 자체를 feature로), (iii) feature-tier
  ablation은 v9 프로토콜 위에서 수행해야 의미 있음.
- **Phase 3 진입 전 사용자 결정 필요**: 2.5 결과(무신호)를 어디까지
  반복 탐색할지 vs 회고 프레이밍으로 전환할지.

## 산출물 인덱스

- 변형별 지표: `results/phase2/{v*,r*}.csv`, 전향: `results/phase2/prospective.csv`
- seed별 예측: `results/phase2/preds/` (로컬), 체크포인트: `data/checkpoints/{변형}/`
- 재현 코드: `src/run_phase2.py`, `src/run_phase2_reg.py`, `src/phase2_data.py`,
  `src/phase2_split.py`, `src/train_loop.py`, `src/prospective_build.py`,
  `src/prospective_train.py`, 분석: `src/analyze_phase2.py`
- causal 데이터: `data/final_df_causal.csv`, 앵커: `data/cohort_meta.csv`,
  전향 윈도우: `data/prospective/windows.npz`
