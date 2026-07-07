# 계획 및 진행상황

목적: 승인된 실행 계획의 canonical 정의와 세션별 진행 로그.
계획 v1은 2026-07-06 논의에서 확정 (B안: 재현 + 진단 + KBO-지향 ablation),
**v2는 같은 날 Phase 1 종료 + ultracode 검증 결과를 반영해 개정** (개정 근거는
진행 로그 마지막 항목). 관련 문서: `kang_repo_audit.md`(코드 감사·환경),
`phase2_findings.md`(검증 36건 + 실행 설계), `reproduction_and_dataset.md`,
`KBO_applicability.md`.

---

## 최상위 목표 (모든 단계의 판정 기준)

- **G1 재현 보존**: 충실 재현 baseline은 동결. 모든 수정은 별도 변형으로
  학습해 baseline과 대조한다.
- **G2 정직한 평가**: 보고 수치는 배치 상황(새 투수, 미래 시즌)의 성능을
  추정해야 한다. 누수·아티팩트·회고 정보는 제거하거나 기여도를 정량화한다.
- **G3 KBO 이전 — 최종 산출물의 형태 확정 (2026-07-06)**: 목표는 "수술일
  예측기"가 아니라 **현재까지의 데이터만으로 계산 가능한 전향적 부상 위험
  지표**(위험도 순위 + 역학적 근거)다. Kang의 회고적 앵커 설계는 그대로
  배치가 불가능하므로(사후 정보로 시간축 정렬), 전향적 재설계(Phase 2.5)가
  KBO 적용의 전제 조건이다.

## 계획 (v2)

### Phase 0 — 데이터 기반 구축 [완료]
- Statcast 2016–2023, 추출·전처리 포팅, 저자 final_df 확보·대조.
- 성공 기준 충족: 코호트 620(101/519), diff 102, sequence 224, X=(620,224,103).

### Phase 1 — 충실 재현 [완료]
- 버그 보존 학습, 분류 5모델 + 회귀 + SHAP 전부 논문 수치 재현
  (성공 기준 F1 ±0.05 충족; 상세는 진행 로그의 대조표).

### Phase 2 — 진단·교정 [완료 2026-07-07 — 결과: phase2_results.md]
ultracode 검증 36건(`phase2_findings.md`) 기반. G1 동결 + 변형별 대조,
ViT+회귀 중심 측정 (변형당 GPU ~45분, 총 ~6시간):
- P2-0 위생(변형별 결과 경로, seed 통일, 이중 소속 8명) →
- P2-1 회귀 투수 단위 split(honest R²) → P2-2 버그 수정 변형(deepcopy,
  ViT shuffle) → **P2-3 fill-artifact 기여도 측정**(상수 꼬리 마스킹 대조 —
  F1 0.73 중 아티팩트 비중 정량화) → P2-4 전처리 변형(F=102, 진짜 선형보간,
  train-only 통계) → P2-5 시간적 홀드아웃(≤2021 학습 / 2022–23 테스트) →
- P2-6 지표·통계(PR-AUC 주지표, valid 기반 threshold, paired Wilcoxon,
  분해능 명시) → P2-7 causal diff 재구축(expanding-window 기준선).
- **성공 기준**: 각 교정의 성능 영향이 방향·크기와 함께 정량화되고, "정직한"
  수치(투수 단위 + 시간 홀드아웃 + 아티팩트 제거)가 확정되는 것. 수치 하락은
  실패가 아니라 산출물이다.

### Phase 2.5 — 전향적 태스크 재정의 [완료 2026-07-07 — 전향 신호 미검출, 이 널 결과가 2.6 재설계의 입력]
- rolling as-of-date: 시점 t까지의 trailing window만 입력, "t 이후 H일 내
  TJS" 라벨(t 이후 수술 기록만 사용), causal 기준선(P2-7), 시간 순서 평가.
- 산출물 형태: 이진 경보가 아닌 **위험도 순위/점수** (낮은 base rate 대응).
- 진입 조건: Phase 2에서 아티팩트 제거 후에도 유의미한 신호가 남을 것.
  남지 않는다면 그 결과 자체를 KBO 컨택 전 필수 정보로 문서화.

### Phase 2.6 — 전향 재설계 (신설·완료 2026-07-07)
- Phase 2.5 널 결과의 원인 분해(informative censoring·라벨 부족·문헌 천장)
  후 재설계 실행: 라이브 라벨(스냅샷의 71건 누락 교정), F=365 위험집합,
  H=90 primary, blackout 민감도, E4 지표(사건 단위 recall·클러스터 CI).
- 결과: workload LR ROC 0.64 / lift 1.6× / event recall@50 29%, blackout
  생존(전향성 확증), 급성 성분(추세·변동성) additive 널 — 독립 검증 통과.
  전체 수치·해석: `phase2_results.md` Phase 2.6절 (canonical).

### Phase 3 — 확장 + KBO-지향 ablation
- +2024 확장 (Roegele 최신 라벨, 재추출 파이프라인 사용). 2025–2026 제외.
- Feature-tier ablation: Tier1(KBO 공개: 구속·구종·회전수) / Tier2(+release
  pos, extension, spin axis) / Tier3(full). **Phase 2 교정된 파이프라인 위에서**
  수행 (누수된 baseline 위의 격차는 무효) + Tier별 동등한 튜닝 예산.
- 창 길이 등은 KBO 실현성 제약(3.3년 밀집 이력 불가) 하에 양 리그 동일 설계.
- KBO 제안서: Phase 2.5 위험 지표 + ablation 격차 + SHAP 근거(상위 feature가
  전부 Tier 2라는 Phase 1 결과)로 구성.
- **[2026-07-07 개정] tier ablation의 새 프레임**: Phase 2.6에서 생존한
  전향 신호의 feature가 전부 공개 계층(투구수·등판일·구속)이므로, ablation은
  "공개 baseline(M0) + Tier-2 증분 검정"으로 재정의 — KBO 요청 논거가
  "핵심 신호가 비공개 계층에 있다"에서 "공개 baseline을 Tier-2가 넘는지
  실측하자"로 이동. 스코프(A 제안서 / B tier ablation / C 추가 실험)는
  사용자 결정 대기.

---

## 진행 로그

### 2026-07-06 (세션 1) — 환경 재현 완료
- uv 기반 Python 3.11.11 venv, torch 2.11.0+cu128 (RTX 5060 Ti sm_120 검증),
  pandas 2.3.3 핀. `scripts/verify_env.py` 9/9 통과. 상세: `kang_repo_audit.md`.
- upstream repo 클론·전 파일 감사. final_df.csv Drive 링크 401 비공개 확인
  → 사용자가 액세스 요청 예정, 파이프라인은 raw 재추출로 진행.

### 2026-07-06 (세션 1, 계속) — Phase 0 진행
- Statcast 2016–2023 다운로드 (~5.5M pitches 예상, 시즌별 parquet).
- `src/extract.py` 포팅 완료. 정독 중 확인된 핵심 사실:
  - 공개 추출 스크립트는 **비부상 그룹의 new_before_tj를 생성하지 않음**
    (부상 그룹에만 생성; `last_game_date` 드롭 등 화석 코드가 실제 생성
    스크립트의 불완전 공개본임을 시사). 논문 Fig 2 설계(마지막 경기 = day 0)로
    갭을 메움 — [GAP-FILL]로 표기된 **추론**.
  - 저자 final_df.csv에는 우리 재구성본에 없는 컬럼 존재 (hitter-outcome diff,
    height/weight/bmi — Prepfortrain의 drop 목록으로 역추정). 전처리 포팅에서
    존재 여부 가드 처리.
  - 5일 bin 채움은 논문의 "linear interpolation"이 아니라 실제로는
    ffill→bfill 후 no-op interpolate = **step-fill** (Prepfortrain L30-33).
  - diff 기준은 논문의 "first-season average"가 아니라 **포함된 전 구간의
    (pitcher, target)별 평균** (Pybaseball_extract L507-515).
- `src/prep_classification.py` 포팅 완료.
- 파이프라인 디버그: pybaseball의 nullable dtype(Int64/Float64)이 boolean mask를
  NA로 오염시켜 코호트 필터가 전멸하는 버그 발견·수정 (`load_raw`에서 numpy
  dtype 정규화 — upstream의 astype(object) 우회와 같은 목적).
- **저자 final_df.csv 확보** (사용자가 액세스 요청으로 공유받음,
  `TJS_Prediction/Raw_data/final_df.csv`에 저장, 136MB):
  - 71,655행 × 133컬럼. **코호트 620 (101/519) — 논문과 정확히 일치.**
  - `new_before_tj`가 양 그룹 모두 존재, 모든 (pitcher,target)의 min=0
    → [GAP-FILL] 추론(마지막 경기 = day 0)이 저자 데이터와 부합.
  - 126 diff 컬럼 = 102 투구 metric + 24 hitter-outcome (+height/weight/bmi)
    → Prepfortrain 드롭 목록에서 역추정한 구조 그대로 확인.
  - **우리 전처리 포트에 통과 → X=(620, 224, 103), y=101/519, NaN 0.**
    모델 입력 feature 수 = 103 (102 diff + new_before_tj_group) 확정.
    Phase 1 학습 입력 확보 완료.
- Statcast 8시즌 다운로드 완료: raw 5,555,493 pitches (논문 5,537,981 대비
  +0.3% — 논문 수치는 정규시즌 필터 이전 기준으로 추정), 정규시즌 5,307,468.
  ParserError 2회는 재시도로 통과 (일시적 Savant 응답 문제).

### 2026-07-06 (세션 1, 계속) — Phase 0 완료: 재추출 vs 저자본 비교
- `src/extract.py` 전체 실행: **656 샘플 (111/545), diff 102개.**
- `src/compare_final_df.py` 결과:
  - **저자 620 샘플이 우리 재추출본에 전부 포함** (theirs-only 0), 클래스 101/519 일치.
  - 시간축(투수별 span·행수) **99.8% 동일** → [GAP-FILL] 공식 검증됨.
  - diff feature: 전역 r 중앙값 0.72이나 **투수별 재중심화 후 r 0.978**
    (95% 컬럼 r>0.95), 투수별 r 중앙값 1.0, flip 불일치 0건.
    → 차이는 per-pitcher 기준선 오프셋(중앙값 ~0.28σ)뿐. 원인 추정:
    Statcast 소급 보정(2026 pull vs 저자 ~2024 pull) + 보간 모집단 차이
    (저자 비공개 스크립트는 hitter metric 포함 등 우주가 다름).
  - ours-only 36명(부상 +10, 정상 +26): 부상 추가분은 Alcántara·Boyd·
    McCullers 등 2023 말~2024 수술자 — repo의 TJ 리스트가 저자 추출 시점보다
    최신 스냅샷이라는 가설(추론)과 부합. 정상 추가분은 Statcast 소급 변경으로
    4연속 시즌 요건 충족이 달라진 것으로 추정(미확정).
- **Phase 0 성공 기준 판정**: 코호트 620(101/519) — 저자본 기준 충족(완전 포함),
  재추출본은 +36(라벨 스냅샷 차이로 정확 일치는 원리상 불가). diff 102 ✓,
  sequence 224 ✓, X=(620,224,103) NaN 0 ✓.
- **Phase 1 방침**: 충실 재현 학습은 저자 final_df.csv로 진행(정확 재현),
  재추출 파이프라인은 Phase 3 확장(+2024, 최신 라벨)에 사용.

### 2026-07-06 (세션 1, 계속) — Phase 1 충실 재현 완료 (SHAP 제외)
- 학습 코드: `src/train_classification.py` (upstream 5개 train 함수를 값 보존
  설정 테이블로 통합, 모델 클래스·data_split은 upstream verbatim import,
  버그 보존: ViT shuffle=False, int64 pos_weight, best 재로드는 ResNet/ViT만),
  `src/train_regression.py` + `src/regression_cnn_fixed.py`(체크포인트 경로만
  수정한 verbatim 복사). 평가는 upstream Visualization 로직의 수치 재현.
- 데이터: 저자 final_df.csv → X=(620,224,103). 분류 10 seeds(100..1000),
  회귀 10 seeds(102..1002). 실행 ~2시간 (RTX 5060 Ti ≈ 논문 L4 속도).

**분류 결과 (injured class, 10-seed 평균) vs 논문 Table 9:**

| 모델 | 우리 F1 / AUC / Acc | 논문 F1 / AUC / Acc | 판정 |
|---|---|---|---|
| ViT | 0.71 / 0.92 / 91.3 | 0.73 / 0.93 / 90.8 | 재현 (±0.02) |
| ResNet | 0.64 / 0.89 / 88.9 | 0.64 / 0.88 / 89.3 | 일치 |
| Transformer | 0.46 / 0.78 / 78.2 | 0.46 / 0.78 / 78.3 | 일치 |
| CNN+LSTM | 0.44 / 0.75 / 74.9 | 0.44 / 0.75 / 74.9 | 일치 |
| LSTM | 0.32 / 0.67 / 65.4 | 0.35 / 0.67 / 63.9 | 재현 (±0.03) |

- 순위(ViT > ResNet > Transformer > CNN+LSTM > LSTM) 그대로 복제.
  seed 간 F1 표준편차 0.04~0.10 — 논문과의 차이는 전부 seed 분산 이내.
- **회귀**: R² **0.783 ± 0.019** (논문 0.79 / repo 0.78),
  100-Day RMSE **94.6 ± 6.9** (논문 95.7). 재현 성공.
- **Phase 1 성공 기준(F1 ±0.05) 충족.** 사전학습 모델(ViT/ResNet) 우위 패턴도
  동일 — 620 표본에서 ImageNet 전이가 지배적 요인이라는 해석 강화.
- 산출물: `results/classification_results.csv`(50행),
  `results/regression_results.csv`(10행), seed별 예측 `results/preds/`(로컬),
  회귀 체크포인트 `data/checkpoints/`(SHAP 재사용).
- **SHAP 재현 (--shap-only, 체크포인트 재사용, ~50분) — Table 10 대조:**
  - top-10 등장 distinct feature 수 **20개 — 논문과 동일**.
  - 평균 |SHAP| 상위 3개가 논문과 **동일 집합**: release_speed_FF(우리 16.5 /
    논문 17.1), spin_axis_FF(15.1 / 14.5), release_extension_SL(13.5 / 13.4).
  - 상위 6개 중 5개가 논문 상위 6개와 일치. 결과:
    `results/regression_top10_features.csv`.
  - KBO 함의 재확인: 상위 feature가 전부 Tier 2(spin_axis, extension,
    release_pos) + 구속 — "중요한 신호가 KBO 비공개 계층에 있다"는 Phase 3
    ablation 가설과 정합.
- **Phase 1 종료.** 분류·회귀·XAI 전부 논문 수치 재현.

### 2026-07-06 (세션 1, 계속) — Phase 2 사전 검증 (ultracode)
- 6차원 코드 감사 + 적대적 검증(에이전트 82개): **36건 생존, 2건 기각**.
  전체 목록·실행 설계: `phase2_findings.md`.
- 핵심: 회귀 split에서 novel test 투수 0명(실측 재현됨), best-weights 참조
  버그, **fill-artifact 지름길**(상수 꼬리 길이가 클래스와 상관 — 신규 발견),
  diff feature의 미래 정보 사용, 회고적 앵커(전향 예측 아님), threshold
  미보정, 코호트 era/내구성 교란.
- Phase 2 실행 설계: G1 동결 + 변형별 대조(ViT+회귀 중심, 변형당 GPU ~45분,
  총 ~6시간). **사용자 승인 대기** — 스코프(P2-0~7)와 전향적 재정의(A5)
  분리 여부 결정 필요.

### 2026-07-06 (세션 1, 마감) — 목표 개정 (계획 v1 → v2)
- 사용자와의 논의로 확정된 사항:
  1. **최종 산출물의 형태**: KBO에 적용 가능한 "부상 지표"란 **현재까지의
     데이터만으로 미래를 예측하는 전향적 위험 점수**여야 함. Kang 설계는
     day 0 = 마지막 경기(사후 정보) 앵커라 그대로는 배치 불가 — 검증 발견
     A5(회고적 앵커)가 한계가 아니라 **재설계 대상**으로 승격됨.
  2. 이에 따라 **Phase 2.5(전향적 태스크 재정의) 신설**. 진입 조건은
     Phase 2의 fill-artifact 기여도 측정(P2-3)과 시간 홀드아웃(P2-5)에서
     아티팩트 제거 후에도 신호가 남는 것. 전향 전환 시 성능은 회고
     수치(F1 0.71)보다 낮아질 것으로 예상하며, 그 정직한 수치가 KBO 컨택의
     근거가 됨.
  3. 전향 가능성의 근거: SHAP 상위 feature(구속 하락, spin axis 수평화,
     release extension 변화)가 수술 전 점진 진행하는 선행 지표 성격
     (논문 추세 r 값들로 지지) — 원리상 가능, 크기는 측정 대상.
  4. 튜닝보다 검증 교정 우선 원칙 재확인 (620 표본·test 부상 20명 환경에서
     튜닝 이득은 노이즈와 구분 불가; ablation 시에만 Tier별 동등 튜닝).
- 운영 방침 확정: git 작업(commit/push/pull)은 사용자 직접. 단계 착수는
  명시적 지시 후 (방법 설명 ≠ 착수 명령). GPU 신규 작업은 승인 전 금지.

### 2026-07-07 (세션 1, 야간 자율 실행) — Phase 2 + 2.5 완료
- 사용자 승인(B안 전체 스코프)으로 P2-0~7 + Phase 2.5 실행. 실행 ~7시간
  (분류 변형 10종 + 회귀 변형 4종 + 전향 학습). **모든 수치·해석의
  canonical 소스: `phase2_results.md`.** 실행 과정: `phase2_worklog.md`.
- 3줄 요약: ① 회귀 신투수 R² = 음수 (논문 0.79는 전부 within-pitcher 보간)
  ② 분류 AUC 0.93 중 ~0.11은 이력길이 아티팩트, **정직한 회고 수치 =
  AUC 0.816** (지름길 제거 × 미래 시즌, v9) ③ 전향(rolling) 예측은 현
  설계에서 무작위 수준 — G3 "부상 지표"의 전제 미충족.
- **Phase 2 성공 기준 충족** (각 교정의 영향이 방향·크기·p값과 함께 정량화,
  수치 하락 자체가 산출물). Phase 2.5는 실행 완료, 진입 조건이었던 "전향
  신호 잔존"은 **불충족** — 계획 v2가 예정한 대로 이 결과 자체를 KBO 컨택
  전 필수 정보로 문서화함.
- **다음: Phase 3 진입 전 사용자 결정 필요** — (i) 전향 설계 반복(IL 라벨
  확장, horizon/창 탐색, 생존분석) vs (ii) 회고 프레이밍(AUC 0.82 위험
  프로파일링)으로 전환. feature-tier ablation은 어느 쪽이든 v9 프로토콜
  위에서 수행.

### 2026-07-07 (세션 1, 계속) — 회귀 split 공정성 논의, compact 대비 정리
- 사용자 질문("회귀는 부상 확정 투수 한정 태스크인데 그래도 투수별 split이
  공정한가")의 논거를 확정, `phase2_results.md` 회귀 절에 기록. 요지: 라벨이
  수술일 역산이므로 배치 대상은 정의상 전원 label-novel → grouped split이
  유일한 배치 정합 평가. 추세 기반 설계는 Phase 3 A안 후보로 명시.
- 상태: GPU 유휴, 실행 중 프로세스·예약 작업 없음. 문서·메모리 compact 대비
  정리 완료. **Phase 3 방향 결정 대기** (A 전향 반복 / B 회고 프레이밍 —
  `phase2_results.md` 마지막 절).

### 2026-07-07 (세션 1, 계속) — Phase 2.6 전향 재설계 확정 (착수는 신호 대기)
- 사용자 방향 확정: **전향 프레임 유지 + 라벨 TJS 고정** (팀원 distance-based
  단기 TJS 트랙과 충돌 방지 — IL/부상 전반으로 라벨 확장 금지).
- ultracode 조사 (에이전트 12: 조사 5각도 + 레퍼런스 검증 5 + CPU 진단 실험 2)
  → 무신호 원인 분해: informative censoring(수술 65%가 prev-30d 규칙에서
  부적격) + 라벨 부족(train 양성 102) + 문헌 천장(진짜 전향 AUC 0.61-0.67).
- fable 재검토 구멍 4개(H×B 커플링, TJ 리스트 2024 경계 절단, fold간 embargo
  부재, 용량 초과→ViT 강등) + codex 교차 검토 6건 수용. 사용자 결정 3건
  (E0 최소판 먼저 / horizon은 supply 표 보고 **최단 우선** / ViT 강등 동의).
  상세·실험맵(E0~E4)·인용 검증 상태: `phase2_results.md` Phase 2.6절.
- **다음 작업 (사용자 "시작" 신호 후, CPU 위주)**: E0a/E0b 라벨·위험집합
  감사 + (H,B) supply 표 → E1+E2 코호트 재구축(windows v2) → 정규화 LR
  baseline → E3 additive ablation(workload→trend→variability) → E4 지표 병행.

### 2026-07-07 (세션 1, 계속) — Phase 2.6 실행 완료 (블록 1–3, 전부 CPU)
- 사용자 "시작" 신호로 착수, 워크플로우 3개(wf_76a925e9 / wf_649b3c96 /
  wf_554f0cb2, 에이전트 8)로 E0~E4 전체 완료. **모든 수치·해석 canonical:
  `phase2_results.md` Phase 2.6절** (블록 1: E0 감사+공급 표 → H=90 primary/
  F=365/라이브 라벨 확정, 스냅샷의 2022-23 수술 71건 누락 발견; 블록 2:
  cohort_v2 + workload baseline ROC 0.640, blackout으로 순환 셧다운 반증;
  블록 3: E3 추세·변동성 additive 널, 독립 검증 통과).
- 3줄 결론: ① 정직 전향 수치 = workload LR ROC 0.64 / PR lift 1.6× /
  event recall@50 29% (blackout 생존 = 진짜 전향 신호) ② 급성 조기경보
  성분은 검출 한계 이하 (회귀 붕괴·blackout 평탄·E3 널 3중 정합) ③ 생존
  신호는 전부 KBO 공개 계층 feature → v1 프로파일러는 트래킹 데이터 불필요,
  데이터 개방 논거는 "Tier-2 증분 검정"으로 전환.
- 데이터 산출물: `data/prospective/cohort_v2.parquet`,
  `game_features_v2.parquet`; 라이브 TJ 라벨 `tj_live_clean.csv`(scratchpad,
  **영속화 필요 시 data/로 복사**). 실험 스크립트·결과는 scratchpad
  (`b1_baseline_v2.py`, `e3_ablation.py`, `E3_RESULTS.md` 등).
- **다음 결정 포인트**: Phase 3 스코프 — (i) KBO 제안서/문서화 정리
  (ii) tier ablation 새 프레임(공개 baseline + Tier-2 증분) (iii) 추가
  실험(H=365 탐색, within-pitcher 고정효과). 사용자 결정 대기.
