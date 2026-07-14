# 계획 및 진행상황

목적: 승인된 실행 계획의 canonical 정의와 세션별 진행 로그. 현재 실행
상태는 파일 끝의 2026-07-13 실행 전 감사 절을 우선한다.
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

## 계획 (역사적 v2 단계 정의; 현재 실행 순서는 파일 끝 최신 절 우선)

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

## 계획 v3 제안 — Phase 3 재정의 (2026-07-07, fable 최종 검토; 사용자 승인 대기)

방향 유지: 전향 프레임, TJS-only 라벨, M0'' canonical, 공개 데이터 v1 프레임.
실행 순서 수정: A+B 병행 → **R(강건성 블록) 선행 → A → B, C 보류**.

### R 블록 — 제안서 대표 수치의 강건성 확보 (CPU, 기계적 실행 가능)
- **R1 rolling-origin 재평가**: test를 2021/2022/2023 각각으로(train=이전
  전부, embargo 동일). 목적: 2022-23 단일 test 반복 조회(선택 편향)와 시대
  효과(COVID 2020 단축, 피치클록 2023) 방어. 판정: fold별 ROC/recall 방향
  일관성(전 fold ROC>0.55 여부만 보고; 미달 시 해석 수정 플래그).
- **R2 role·age 공변량**: M0'' + 선발 비중(GS/G 또는 경기당 투구수 프록시)
  + 나이. 목적: "만성 사용량 낮음=고위험"이 불펜/연령 프록시인지 분해 +
  성능 개선 가능성. paired delta 보고. pc_chronic 계수가 role 추가로 크게
  죽으면 해석문을 "역할 포함 프로파일"로 수정.
  **[2026-07-08 role 부분 완료]**: start_share additive가 dPR CI 0 상방
  배제(양 H) → **M-role(M0''+start_share) 잠정 canonical** (R1 재확정
  조건). Simpson 확증: 역할 간 prevalence(선발 2배) vs 역할 내 사용량
  감소 위험 — pc_chronic 계수 −0.185→−0.38. 상호작용·역할 내 표준화
  불필요. **R2 잔여 = 나이만.** 상세: phase2_results.md 블록 5.
- **R3 회고 v9 라이브 라벨 재실행**: v9 코호트는 구 스냅샷 라벨 기반 —
  누락 71건 일부가 정상군 오염 가능(0.816은 과소측정 추정). extract를
  라이브 리스트로 재실행 → v9 재측정, 제안서 대표 수치 확정.
- **R4 KBO 라벨 소스 스코핑** (조사): 공개 KBO TJS 기록 존재 여부. 부재 시
  제안서 프레임 = "구단 내부 의무 기록으로 자체 검증 가능"(구단 카드).

### A 제안서 (R 반영 후) — 양면 논리
회고 0.82(R3 확정치) = 투구 역학에 신호 실재 / 전향 0.62·recall 25-38%
(공개 feature만) = 배치 가능한 v1 / 급성·Tier-2 미검출 = 검정력 한계 →
**요청: 다년·다구단 데이터로 (i) 검정력 확보 (ii) Tier-2 증분 검정.**
- **이전성 주의 (2026-07-07 논의)**: workload 신호의 부호 구조는 MLB
  사용량 배분 생태계(healthy-worker, 관리된 부하, 빠른 수술 결정)의 산물.
  KBO는 ① 선수층이 얇아 전조(수술 전 사용량 감소)가 약할 수 있고 ② 혹사
  분산이 커 MLB에서 눌린 인과 방향(급성 부하↑→위험↑)이 반대 부호로 잡힐
  수 있으며 ③ 수술 시점 관행(재활 선호·병역)이 달라 라벨 타이밍 분포가
  다름(정성 서술 — R4에서 검증). → **이전 대상은 가중치가 아니라
  방법론(설계+진단)이고 KBO 재적합이 전제** — 이것 자체가 데이터 협력
  필수 논거. KBO는 이벤트 수가 적어 다시즌 축적 + event recall 위주 평가.

### B tier ablation (A와 병행 가능)
M0''+role+age 위에 Tier-2 만성 pitch-shape 수준(시즌 평균 spin axis·
extension·release 등) 증분. **사전 기대 명시: 현 검정력(events 52-56)에서
미검출이 기본 가설** — 어느 결과든 A의 논거로 사용 가능.
- **B' 확장 (2026-07-08 논의 — "사용량 신호는 구단이 이미 아는 정보"
  반론 대응)**: 정보원 분리 3종 추가. (i) **콘텐츠-only 모델** — 사용량·
  활동 feature 전부 제외, 투구 내용만으로 단독 성능 (E3 널은 "workload 위
  증분 없음"이지 콘텐츠 단독 무신호가 아님 — 미검정) (ii) **사용량 정상
  하위집단 평가** — 사용량 감소 없는 윈도우 한정 수술자 포착률 = 구단
  행동에 선행하는 신규 정보의 크기 (iii) **lead 분석** — 포착된 수술자가
  포착 시점에 이미 사용량 감소 중이었는지 분해. 결과별 제품 스토리:
  콘텐츠 신호 있음 → "공이 먼저 말한다" + Tier-2 가치 직접 입증 / 없음 →
  제품 가치 = 타 구단 투수 평가(FA·트레이드 실사; 정보 비대칭 해소) +
  리그 전체 체계적 모니터링·캘리브레이션으로 프레이밍. horizon 연장은
  정보원 문제를 해결하지 못하므로 이 반론의 해법이 아님(C 유지).

### C 보류 (학술 후속)
H=365 탐색, within-pitcher 고정효과, Dillon식 최종 등판 이상탐지.

### 실행 가드레일 (fable 부재 세션용)
- **재검토 없이 변경 금지**: 라벨 정의(TJS-only)·eligibility(F=365, ≥20경기)·
  H=90 primary·M0'' canonical·평가 프로토콜(train+valid 적합, 투수 클러스터
  부트스트랩 seed 0, event recall, paired delta).
- 신규 실험은 `results/phase26/scripts/` 프로토콜 재사용 + **M0 재현 체크
  선행** + paired delta 보고 형식 유지.
- 스토리(해석) 수정이 필요해지면 근거를 문서에 기록하고 사용자 승인 후 진행.

### 2026-07-10 — B/B' 블록 완료: tier × 모델 ablation (전부 널)
- 사용자 결정: **제안서 작성은 사용자가 직접**(Claude 몫 아님), 실행 순서 =
  B(Tier-2 증분) + tree 모델 비교 먼저. tree 시도 합의, "가능성 보이면
  hyperparameter 탐색" 조건부.
- 설계·수치 canonical: `results/phase3/B_TIER_MODELS.md`(상세) +
  `phase2_results.md` 블록 7(판정). {F1=M-role, F2=+Tier-2 12개,
  F3=콘텐츠-only} × {LR 동결 프로토콜, HistGBM valid-전용 그리드},
  H90/150, paired bootstrap 1000 공유, M-role 재현 게이트 1e-7 통과.
- 판정: ① **Tier-2 증분 널 3중 확인** — 달력창(NaN 35-49% 희석 우려) →
  경기창(최근 5/15경기, NaN<8%) → 관측 부분집합, 전부 CI 0 포함/음수;
  +0.05 초과 증분 배제. ② **tree 널** — 동일 feature HGB≈LR, feature 확장
  시 유의 악화(−0.07~−0.08 EXCL0); 경계 밖 확장 그리드 48설정도 무효
  (valid→test 전이 실패 = selection noise) → **조건부 hyperparameter 탐색은
  조건 불충족으로 종료**. ③ 콘텐츠-only ROC 0.49-0.55 — 사용량 정보 대체
  불가. ④ 양성 없음 → rolling-origin 생략(사전 규칙). **M-role 유지.**
- **B' (ii)(iii) 완료 (같은 날, 사용자 "시작하자")**: canonical
  `results/phase3/B_PRIME_LEAD.md` + phase2_results.md 블록 8. 핵심: ①
  급성 감소 flag 단독은 window 수준 무판별(base rate 비 ~1×) — 신호는
  만성 사용량×역할 ② H150 포착의 60%(9/15)가 사용량 정상 시점, lead
  중앙값 72일; 정상 하위집단 내 ROC 0.663 [0.58-0.75] ③ H90은 셧다운
  중첩(4/13 정상) → "150일 조기 프로파일 + 90일 임박 확인" 2단 포지셔닝
  가능. 포착 n 작아 분수는 점추정. B/B' 블록 전체 종료.
- G3 논거 확정: "MLB에서 Tier-2 증분 입증"은 불가 → "MLB 규모 검출 한계
  이하, KBO 내부 라벨과 결합해야 검정 가능(데이터 접근 = 검정력)"으로 전환.

### 2026-07-10 (계속) — 성능 진단 + 다음 스텝 설계안 (ultracode, 사용자 확정 대기)
- 사용자 요청: "더 나은 성능 가능할 것" → 학습 로그·코드 진단 + 문헌·
  GitHub·웹 종합 조사로 다음 스텝 구체화 (코드 수정 금지, 설계안만).
- 실행: 6-agent 워크플로우(진단 3 + 조사 3, opus) → fable 종합 + 핵심
  수치 2건 재검증. **canonical: `results/phase3/NEXT_STEPS_DIAGNOSIS.md`**.
- 진단 요지: 버그 없음. 하드 리밋(사건 52-56 플로어, 불펜 구조적 불가 —
  포착 전원 선발, 문헌 천장 0.61-0.67) + 교정 가능 depressor 2건(조기
  시즌 workload 구조적 0 — 4월 82% 퇴화, 프로브 +0.01-0.015; vel_trend
  결측→0) + 2023 약세(0.60 vs 2022 0.69).
- 설계안: P0-1 2024 확장(test 사건 +50%, 위협 검증 완료, 다운로드 필요) →
  P0-2 M-role v2 구조 교정 → P1 미니 티어(prior-TJS flag — 검증 결과
  **보호 방향 0.55×**; pitch-mix 단독 재검정; rest 구조) → P2 discrete-time
  hazard 재정식화. 기각: 시퀀스 DL·NCC·HGB 추가·Tier-2 재방문.
- ~~대기 결정 3건~~ → **사용자 승인 (2026-07-10)**: 순서대로 실행, 천장
  사전 설정 없음, 2024 다운로드 승인.

### 2026-07-10 (계속 2) — P 블록 실행 완료: canonical v4 (hazard, ROC ~0.70)
- canonical: `results/phase3/P_BLOCK_RESULTS.md` + phase2_results.md 블록 9.
- P0-1: statcast 2024 다운로드(760k 투구) → v3 데이터(slim/gf/cohort,
  ≤2023 부분집합 byte-동일 게이트) → test 2022-24, 사건 75/80 (+44%).
- P0-2: **M_sa 채택** (+prior_pc_rate, ncg_log, vt_missing; H150 dROC
  +0.017 EXCL0, rolling 6/6 양, ridge 동일). M_bf 백필은 재검 기각.
- P1: 미니 티어 전부 미채택 (prior_tjs는 방향 양 + evrec@50 +4, CI 포함 —
  사건 확대 시 1순위 재검정 후보로 기록).
- P2: **hazard supermodel 채택 = canonical v4** — ROC 0.701/0.708,
  H-정합 보장(이진 쌍 47% 위반 해소), plain LR 확률 보정. role-strat 미채택.
- 대표 수치: 3년 평균 ~0.70 (연도 0.62-0.82). 산출물: results/phase3/
  p0_*/p1_*/p2_* csv + scripts/. 데이터: data/prospective/*_v3.parquet.
- 코드 변경: src/download_statcast.py SEASONS → 2024 포함 (기본 동작:
  기존 시즌 스킵). 나머지는 전부 신규 스크립트 (기존 코드 무수정).
- 남은 것: 제안서(사용자 직접) 수치 갱신 반영 + 필요 시 hazard 모델의
  운영 산출물(월간 순위 시트) 생성.

### 2026-07-10 (계속 3) — 사후 검증 + A1/B1 확장, 개선 캠페인 종료
- 사후 적대 검증(사용자 "문헌 초과 의심"): 라벨 실재·as-of 무결·M_sa의
  2024 비의존성 통과. **핵심 각주 = 추정량 차이**: window-pooled 0.70 vs
  문헌 비교용 스냅샷 ~0.69; 2022-23은 문헌 대역 안. "문헌 초과" 주장 금지.
  P_BLOCK_RESULTS.md 검증 절.
- A1: 2025 test 추가(라벨 성숙 재점검 통과) → **데이터 v4, test 사건
  94/100**. 2025는 보통 해(0.63-0.65) → 2024 outlier 확정.
- **최종 대표 수치: hazard, test 2022-25, ROC 0.689/0.693 [0.64-0.74],
  recall@50 ~28%, 통상 연도 0.63-0.68.**
- B1 경기 내 구속 감쇠 기각(널). prior_tjs 3차 미채택(방향 양 지속,
  CI 0 포함 — KBO/시즌 축적 시 1순위). A2/B2 보류(기대값·검정력 부족).
- **개선 캠페인 종료 판단**: 현 데이터 계층의 레버 소진. 다음은 KBO 이전
  패키지 (방법론+파이프라인+진단, 계수 재적합 전제).

### 2026-07-13 — codex 외부 감사 수용, 계획 수정 (A0 교정 + A1 불펜 대기)
- codex 감사의 검증 가능 주장 6건을 fable이 전부 재계산으로 확인
  (`results/phase3/scripts/v_codex.py`) → **전부 사실**. 핵심 수용:
  ① 2025 포함은 E0A 라벨 신뢰 종료(2024-12-31) **위반 — 내 오류** (시트
  최신 날짜 ≠ 수집 완결성). 2022-25 헤드라인 철회, **정정 canonical =
  t+H≤2024-12-31, H90 0.701 [0.643,0.759] / H150 0.696 [0.645,0.746]**,
  "조건부 backtest" 표기. 2025 창 = 동결 모델의 향후 전향 확인 세트(라벨
  성숙 ~2027 중반). ② hazard "이중계상 제거·확률 보정" 과장 철회 (양성
  구간 252 vs distinct 79, EPV ~7.9; slope ~1.9 → validation 재보정 필요)
  ③ 무작위 lift 2-3× → **1.55-1.76×** (MC 2000회, 유의 P≤0.012)
  ④ "불펜 구조적 불가" → **경보 배분 문제로 재규정** (RP-내 ROC 0.65,
  top-50 RP 포착 1건) ⑤ seen 0.71 / **novel 0.66** → KBO 기대치는 novel
  기준 ⑥ 회고 F1@0.5는 임의 작동점 — 인용 금지.
- **수정 계획 (codex A안 채택)**:
  - **A0 평가 교정** — 완료: reliable_end 규칙·정정 수치·calibration 실측·
    MC lift·role/novel subgroup·문서 정정. **잔여**: (i) nested
    rolling-origin으로 선택 낙관 추정(등록된 선택 규칙 자동화 범위)
    (ii) validation-only 재보정 파이프라인 (iii) hazard 사건-가중/cluster
    민감도 (iv) 이후 **모델 완전 동결**.
  - **A1 불펜 mini-block** (사용자 승인 대기): GS 기반 역할 재정의(raw에서
    선발 식별 계산), RP 시간척도 feature 소수(2/3/7/14/28일·연투·back-to-
    back), Cohen 2022 release-side slope×RP 1회 사전 등록, 경보 quota는
    validation 선택 Pareto. Moore 2026 torque 모델은 코드 확보 gate.
  - **KBO 이전**: 이전 대상 = 정의·계산법·검열/보정 절차·role-aware 평가·
    경보 정책 (계수 재적합). 기대치 novel ~0.66. 데이터 계약 최소 목록
    (TrackMan raw+메타, 실제 역할/로스터, 숨은 투구량, UCL 진단·수술 상세,
    prior TJS, 연결 ID). TJS/UCL primary 유지, arm IL은 (라벨 아닌)
    auxiliary feature로만 검토 — 사용자 확인 필요.

### 2026-07-13 (계속) — A0·A1 사용자 승인, 사전 등록 실행 사양
- **사용자 승인**: A0 잔여 + A1 불펜 mini-block 모두 권장안대로 진행.
  **arm-IL feature는 설명 제공됨** (요지: 라벨 불변, 팔꿈치 IL "이력"을
  입력으로 — prior_tjs 계열; MLB는 새 데이터 소스 필요라 A1 범위 밖,
  결정은 KBO 설계 시점) — **사용자 결정 대기**.
- **A0 잔여 사양 (사전 등록; 순서대로, canonical 수치는 불변·보고만)**:
  1. nested rolling-origin: fold Y∈{2022,2023,2024}(t+H≤2024-12-31 유지),
     fit=year<Y에서 등록된 선택 경로(M-role/M_bf/M_sa 중 paired 점추정
     최선 선택 + hazard 비열등 채택)를 자동 재실행 → fold-내 선택 성능
     vs 고정 M_sa+hazard 성능 차 = 선택 낙관 추정치.
  2. validation 재보정: fit set 내 5-fold out-of-fold 예측으로 로지스틱
     재보정(slope/intercept) 적합(valid 2021 단독은 사건 15개라 CV 사용)
     → test에 적용, calibration-in-the-large/slope/Brier/reliability 보고.
  3. hazard 민감도: (i) 사건-가중(양성 구간 가중 1/landmark 수) (ii) s
     범주형 baseline — paired 점추정 비교.
  4. **동결 [후속 정정으로 대체됨]**: 위 완료 후 `results/phase3/FROZEN_MODEL.md`에 스펙+계수
     고정 기록, "2025-26 라벨 성숙(~2027 중반) 시 1회 전향 확인" 명시.
     이후 MLB 쪽 모델 변경 금지.
- **A1 불펜 mini-block 사양 (H150 primary, H90 secondary)**:
  1. 역할 재정의: raw에서 game_pk별 팀 첫 투수=GS 계산 → 5분류
     (starter / opener·bulk / swing / short RP / long RP; 규칙은 실행 시
     문서화, start_share 프록시와 대조).
  2. RP 시간척도 feature 소수(≤6): 2/7/14일 투구수·등판수, back-to-back·
     3-in-4 횟수, 직전 등판 투구수의 개인 baseline 대비 급증.
  3. Cohen 2022 검정 1회: handedness-정규화 release_pos_x의 1-3년 기울기
     × RP 상호작용 (회고 case-control 출처 — 널이면 즉시 종료).
  4. 모델: pooled vs role-interaction vs SP/RP 분리(shrinkage) 비교.
  5. 경보 quota: validation에서 SP/RP 배분 선택 → test 1회. **채택 기준 =
     같은 총 예산(50)에서 RP 포착 증가 AND 전체 recall 손실 ≤ 사전
     허용치(2건) Pareto 개선** (RP ROC 단독 아님). Moore 2026 torque는
     코드 확보 전 제외.
- 실행은 compact 후 세션에서 이 사양대로 기계적으로 진행 (재설계 금지).

### 2026-07-13 (계속 2) — arm-IL feature MLB 실험 승인 (A-IL 블록), 동결 순서 조정
- **사용자 승인**: arm-IL 이력 feature를 KBO 대기가 아니라 **MLB에서도
  검정** (데이터 소스 확보 어렵지 않다고 판단). 실행은 compact 후 사용자
  "시작" 지시 대기.
- **순서 조정 (논리적 필연)**: 모델 동결(구 A0-4)은 A0-3 직후가 아니라
  **모든 승인 블록 종료 후**로 이동. 확정 순서:
  **A0(1-3: rolling-origin 낙관·OOF 재보정·hazard 민감도) → A1(불펜) →
  A-IL(arm-IL feature) → 동결(FROZEN_MODEL.md)**.
- **A-IL 블록 사전 등록 사양**:
  1. 데이터: MLB StatsAPI transactions 엔드포인트(무료)에서 2016-2024 IL
     트랜잭션 수집, 텍스트에서 팔꿈치/전완 키워드(elbow, forearm, UCL 등
     — 키워드 셋은 수집 시 문서화) 파싱. 파싱 규칙·커버리지 감사 필수
     (연도별 IL 건수 sanity check).
  2. feature (as-of t, 엄격히 t 이전, ≤3개): 최근 2년 팔꿈치-IL 등재 수,
     마지막 팔꿈치-IL 경과일(log/cap), 최근 2년 임의-IL 등재 수(대조용).
  3. 검정: 당시 baseline(M_sa 또는 A1 채택분) 위 additive paired, 성숙
     test(t+H≤2024-12-31), EXCL0 시 rolling-origin. **신규 정보 분해
     필수**(B'식): 팔꿈치-IL은 구단이 이미 아는 정보이므로, 포착 사건 중
     "IL 이력 없이 잡힌 것"의 비율을 병기해 조기경보 가치를 분리.
  4. 주의: 수술 직전 팔꿈치-IL은 사실상 진단 공개 — lead 분포를 반드시
     보고 (near-t IL 의존 포착과 순수 조기 포착 구분).

### 2026-07-08 — 계획 v3 승인, R 블록 완료 (R3 GPU 재실행 진행 중)
- **R1 rolling-origin: 기준 (a)(b) 모두 PASS.** 전 12 cell(모델×연도×H)
  ROC>0.55(최저 0.572), M-role의 dPR·dROC 점추정 6/6 fold 양(+). 2023
  (피치클록)에서 workload 모델이 가장 약하고 M-role 이득이 가장 큼 =
  아티팩트의 반대 방향. **M-role(M0''+start_share) canonical 확정.**
  단서: fold당 이벤트 17-29건이라 단일 fold 유의성은 없음 — 방향 일관성
  기준의 통과이며, 이는 사전 명시된 기준임.
- **R2 age: 미채택.** 짝지은 CI 모두 0 포함 + 점추정 부호가 H 간 반전.
  단 ① 단변량 나이 기울기는 실재(젊을수록 발생률 3-4×, 생존자 효과 해석)
  ② age가 pc_chronic 계수를 안 움직임 → "만성 사용량=나이 프록시" 가설
  기각. 필요 시 해석용 공변량으로만.
- **R3 prep 완료 + GPU 재실행 착수**: extract.py --tj-csv/--out 플래그
  (기본 동작 불변, upstream 무수정), 라이브 재추출 652 샘플(부상 111→124).
  **핵심 발견: v9 코호트 영향은 "71건 누락"이 아니라 2024 수술 라벨
  성숙** — 신규 부상 13명 중 11명이 2024 수술자(스냅샷엔 미래라 없었음),
  71건 중 Kang 코호트 진입은 2명뿐(나머지는 마이너). v9 temporal test
  부상 48→61(+27%, 전부 2023 anchor). **GPU 재실행 완료 (meta 복구 검증됨)**:
  seed-paired ROC v9o_snap 0.675 → v9o_live **0.601** (p=0.002, 10 seed
  전부 하락). 저자 v9=0.816. **회고 대표 수치 하향 정정: 정직 회고
  ROC ~0.60** (교정이 수치를 낮춤 — 신규 양성이 2024 수술자로 더 어려움).
  회고(~0.60)·전향(~0.64) 대역 수렴 → "붕괴" 서사 폐기. 상세:
  phase2_results.md 블록 6.
- **R4: partial_news_based.** 공개 KBO TJS 시트 부재 확증(위키·스탯티즈·
  MyKBO 전무), 뉴스 재구성은 1군 유명 투수 편향(연 5-15건 추정, 수술일
  정밀도 혼재). **제안서 핵심 카드: KBO 부상자명단 제도(2020~)가 이미
  '부상 일자·부위·진단서'를 구단→KBO 경로로 수집·보유(비공개) → 요청은
  "새 데이터 수집"이 아니라 "이미 존재하는 내부 라벨에의 통제된 접근".**
  공개 라벨은 하한 검증 세트로만.
- 다음: v9o 완료 → R3 수치 확정(회고 대표 수치) → A 제안서 + B/B'.
- **R 블록 종료 상태 (2026-07-08)**: 산출물 `results/phase3/`
  (R1_ROLLING/R2_AGE/R3_PREP/R4_KBO_LABELS.md + csv + scripts). 코드 변경
  잔존: `src/extract.py`(--tj-csv/--out 플래그, 기본 동작 불변),
  `src/run_phase2.py`(LIVE_CSV + v9o_snap/v9o_live 변형). 데이터:
  `data/final_df_live.csv`, `data/tj_live_for_extract.csv`,
  `data/cohort_meta_{snapshot,live}.csv`(cohort_meta.csv는 스냅샷으로 복구됨).
  **run_phase2 meta 주의: temporal 변형은 data/cohort_meta.csv 단일 참조 —
  v9o_live 재실행 시 cohort_meta_live.csv로 교체 후 복구 필요**(anchor 로직
  미패치, 가드레일 준수).
  **사용자 대기 결정 2건**: (a) 제안서 프레이밍 — 정직 우선(공개 feature
  ~0.6 안정 신호 리드; 권장) vs 이중 트랙(논문 재현 0.82 방법검증 인용 +
  자체 ~0.60 병기). (b) 다음 실행 — B/B' tier ablation(Tier-2 증분 +
  콘텐츠-only)을 제안서 전에 돌릴지. GPU 유휴, 예약 작업 없음.

### 2026-07-07 (세션 1, 계속) — 다중공선성 정돈 (블록 4)
- 사용자 지적("feature 상관 큼")으로 감사 실행: workload 블록 VIF 8.6-9.7
  (E3 신규 feature는 깨끗 ≤1.6 → E3 널 강화). A안 채택 → M0'(중복 3개 제거,
  성능 동등·recall 개선) → **M0''(만성 수준+급성 이탈 재매개화, max VIF
  1.64)를 canonical baseline으로 확정**. 계수 해석 가능해짐: "만성 사용량
  낮음 + 최근 급감 = 고위험". 상세: `phase2_results.md` 블록 4.

### 2026-07-13 (최신) — Codex 실행 전 감사 및 권한 동기화
- **실행 없음**: 코드·데이터 수정, 파일로 저장한 다운로드, 승인 블록 실행
  모두 하지 않음. 사용자의 최신 지시에 따라 `.md`만 정정했다. 이후 `.md`
  외 작업은 별도 `시작` 지시가 필요하다.
- 큰 순서 **A0(1-3) → A1 → A-IL → 동결**은 유지할 가치가 있으나, 현재
  사양은 아직 기계적 실행 준비가 끝나지 않았다. 사용자 설계 확정 전 보강할
  항목: A0 inner split/비열등 margin·동률 규칙, calibration OOF의 시간·투수
  분리, A1 역할 임계값·정확한 6개 feature·shrinkage·quota grid, A-IL
  regex/episode dedup/coverage gate·near-t 승격 기준.
- A0-1은 현재 정의대로면 전체 adaptive selection optimism이 아니라 사전
  고정 3후보의 nested stability audit이다. 낙관은 `inner 선택 추정치 - 선택된
  모델의 outer 성능`으로 별도 정의해야 한다. A0-2는 H90/H150 공동 정합을
  깨지 않는 grouped/temporal cross-fit, A0-3은 양성 가중 외 cluster-level
  full-refit 민감도를 사전 고정해야 한다.
- A1에서 "H150 primary"는 **불펜 블록 채택 판정**에 한정하고 전체 시스템은
  H90/H150 공동 출력으로 유지한다. validation 2021의 distinct 사건은 17개라
  단일 최적 quota보다 pre-test rolling fold의 안정 영역을 사용해야 한다.
- A-IL은 placement/transfer/activation을 episode로 합치고, transaction
  `date`를 정보시점으로 사용한다. near-t 공개 진단만으로 통과하지 않도록
  30/60/90일 lag 또는 blackout 민감도와 조기신호 승격 기준을 먼저 정한다.
- **라벨 경계 민감도 추가 확인(읽기 전용)**: E0A가 명시한 full 2-year
  safety end `2024-06-30` 적용 시 hazard ROC는 H90 0.660 [0.596,0.726]
  (67 사건), H150 0.665 [0.603,0.726] (56 사건). primary 0.701/0.696은
  유지하되 2024 강세 연도/경계 의존성을 함께 보고해야 한다. safety 경계의
  novel ROC는 0.572/0.557이므로 KBO 기대치를 0.66으로 고정하지 않는다.
- **전향 확인 표현 정정**: 2025 성능은 이미 조회되었으므로 향후 라벨 갱신도
  untouched test가 아니다. 2025는 label-refresh robustness set으로만 쓰고,
  진짜 전향 평가는 동결 뒤 처음 timestamp한 미조회 decision cohort에서 한다.
- **A-IL 소스 소표본 점검(파일 저장 없음)**: MLB StatsAPI 2016/2024 표본에서
  부위 텍스트는 대체로 존재했지만 retroactive `effectiveDate`, 부위 누락,
  transfer/activation 중복, 수술 후 회복 문구가 함께 확인됐다. 정보시점은
  transaction `date`, placement episode만 사용하고 파싱 coverage/정확도
  gate를 성과 조회 전에 동결해야 한다.
- 상태: **사용자 검토 및 설계 확정 대기**. 위 보강 전에는 승인 블록을
  실행하지 않는다.

### 2026-07-13 (계속 3) — codex 2차 감사 수용 + 실행 사양 v2 확정 (fable)

- **codex 수치 검증 (fable 재계산, scratchpad read-only — 프로젝트 파일
  없음)**: safety 경계 `t+H≤2024-06-30`에서 H90 0.6595 [0.596,0.726]
  사건 67 / H150 0.6652 [0.603,0.726] 사건 56, novel 0.572/0.557,
  RP-내 0.601/0.621, lift 1.57×(P=.043)/1.43×(P=.072) — **codex 보고와
  반올림까지 전부 일치, 수용 확정**. 방법론 지적 6건(A0-1 정의, OOF
  grouping, cluster 민감도, A1 자유도, A-IL leakage, 2025 재분류)도
  전부 타당 판정.
- **인용 규칙 (확정)**: primary 0.701/0.696은 항상 safety 경계
  0.660/0.665와 병기. KBO 기대치는 novel 대역 **~0.56–0.68**로 인용
  (0.66 단일값 금지). "H150 primary"는 A1 블록 채택 판정에 한정, 시스템
  출력은 H90/H150 공동 유지.
- **아래가 사전 등록 사양 v2 — "2026-07-13 (계속)" 절의 사양과 충돌 시
  v2가 우선. 사용자 "시작" 지시 후 이대로 기계적 실행 (재설계 금지).**

**A0-1 (개명: nested 3-후보 안정성 감사)** — 전체 adaptive search 낙관이
아니라 사전 고정 3후보 선택의 안정성 감사이며, 결과는 **전체 낙관의
하한**으로만 해석·보고한다.
- fold Y∈{2022,2023,2024}; outer 평가 = 연도 Y 창 (t+H≤2024-12-31 유지).
- inner 분할: fit = 연도 ≤ Y−2, inner-valid = 연도 Y−1.
- 등록된 선택 경로: 후보 {M-role, M_bf, M_sa} 이진 LR → inner-valid
  paired 점추정 ROC(H90·H150 평균) 최고 선택; **동률 규칙**: 차이 <0.002면
  feature 수 적은 쪽. 이후 hazard form은 inner-valid ROC(hazard) ≥
  ROC(이진) − 0.005(**비열등 margin**)이면 채택.
- **낙관 정의 = fold별 [inner-valid 추정 ROC(선택 모델) − outer 연도 Y
  ROC(같은 모델)]**, 3-fold 평균 보고. secondary: outer(선택) −
  outer(고정 M_sa+hazard). 보고 전용, canonical 불변.

**A0-2 (재보정)** — row-random 5-fold 금지.
- cross-fit: fit set(2017-21)에서 **투수-grouped 5-fold** OOF person-period
  hazard 예측 생성.
- 재보정 위치 = **hazard 구간 수준 공동 재보정**: OOF 구간 예측 logit에
  로지스틱(a+b·logit h) 적합 → 재보정 h̃로 P(H)=1−∏(1−h̃) 재계산 —
  P90≤P150 정합 유지. H별 marginal Platt은 참고용 민감도(정합 깨질 수
  있음 명시), temporal(leave-one-year-out) cross-fit도 민감도로 병행.
- 보고: calibration-in-the-large / slope / Brier / decile reliability를
  mature test와 safety 경계 각각에서.

**A0-3 (hazard·cluster 민감도)** — 기존 (i) 사건 가중(양성 구간
1/landmark 수) (ii) s 범주형 baseline에 추가:
- (iii) **cluster-level full-refit bootstrap**: fit-set 투수 단위 재표집
  B=200, 매회 hazard 재적합 → 고정 mature test ROC 분포 (적합 불확실성
  반영 CI).
- (iv) **수술당 단일 landmark 민감도**: fit set에서 각 (투수, 수술)의
  양성 구간을 수술에 가장 가까운 landmark 1개만 유지(규칙 고정:
  min(수술일−t)), 재적합 → paired 점추정. 전부 보고 전용.

**A1 (불펜) 사양 v2**
- 역할: **모델링용 3분류** — trailing 365d GS share(등판 중 GS 비율)
  ≥0.5 SP / ≤0.2 RP / 나머지 swing. 5분류(starter/opener·bulk/swing/
  short RP/long RP)는 기술 통계·오류 분석 전용 (opener는 "팀 첫 투수 &
  투구 <30" 근사 — 서술용, 모델 미사용).
- feature **정확히 6개 고정**: pitches_7d, pitches_14d, appearances_14d,
  b2b_count_30d(연속일 등판 횟수), three_in_four_count_30d,
  last_outing_spike(직전 등판 투구수 − trailing 90d 등판당 중앙값).
  2일 창은 제거(days_since_last와 중복).
- 모델 비교: pooled vs role-interaction vs SP·RP 분리 — 분리 모델은
  ridge C=0.1 고정(소사건).
- Cohen 2022 release-drift×RP 1회 사전 등록 검정 유지 (널이면 종료).
- 경보 quota: 단일 최적 대신 **안정 영역 규칙** — RP 예약 slot grid
  {0,5,10,15,20}/50을 pre-test rolling fold Y∈{2019,2021}(fit<Y;
  2020은 단축 시즌으로 제외)에서 Pareto 계산, **모든 fold에서 총 사건
  손실 ≤1 (vs quota 0)**을 만족하는 최대 RP quota 채택; fold 간 불일치
  시 작은 쪽(보수). test 1회 적용, 채택 게이트 유지 = RP 포착 증가 AND
  총 recall 손실 ≤2건. Moore 2026 torque는 코드 확보 전 제외 유지.

**A-IL 사양 v2**
- 정보시점 = transaction **`date`** (retroactive effectiveDate 금지).
- episode 구성: placement→transfer→activation 체인을 1 episode로 병합
  (activation 또는 60일 무활동으로 종결), 중복 placement 제거, 마이너
  트랜잭션 제거(MLB 소속 필터), "recovering from Tommy John surgery"류는
  신규 팔꿈치 episode에서 제외하고 post-TJS 상태 플래그로 별도 분류.
- **성능 조회 전 동결 게이트**: ① 키워드 regex 셋 고정·문서화(elbow/
  forearm/UCL/ulnar/Tommy John, 대소문자 무시) ② 연도별 episode 수
  sanity + 무작위 40건 수동 검수 정밀도 ≥90%. 게이트 기록 후에만 평가.
- **blackout 민감도 필수**: t 직전 {30,60,90}일 내 IL 이벤트 제외
  재계산. **승격 기준(사전 등록): 60일 blackout에서도 paired 개선
  방향(점추정 양) 유지 시에만 canonical 후보** — 아니면 "공개 진단 후
  triage 신호"로 분류, canonical 제외·별도 문서화.
- 신규 정보 분해(B'식) + lead 분해 필수 — 기존 유지.

**동결·전향 확인 (정정 반영)**
- 동결 = 전 블록 종료 후 FROZEN_MODEL.md에 스펙+계수+재보정+quota 정책
  고정, **그리고 당월 decision cohort 점수를 timestamp 저장** — 이
  시점부터가 진짜 전향 평가의 시작. 2025 창은 label-refresh robustness
  set으로만 사용(이미 조회되어 untouched 아님).
- 상태: **사양 v2 사전 등록 완료. 실행은 사용자 "시작" 지시 대기.**

### 2026-07-13 (실행) — 사용자 "시작" 지시, A0·A1 완료

- **A0 완료 (보고 전용, canonical 불변)** — `results/phase3/A0_RESULTS.md`:
  - A0-1 nested 3-후보 감사: fold별 선택 불안정(M-role→M_sa→M_bf), 3-fold
    평균 낙관 **−0.013**(체계적 낙관 없음, 연도 노이즈 ±0.07 지배), 고정
    M_sa+hazard가 전 fold에서 fold별 선택보다 우수. 하한으로만 인용.
  - A0-2 재보정: grouped-OOF 재보정 계수 **a −0.164 / b 0.967 ≈ 항등** —
    test slope ~1.9의 원인은 과적합이 아니라 **2017-21→2022-24 유병률
    shift**로 fit-set 재보정으로 교정 불가. 절대 확률 = 배치 시점 재보정
    필수 규칙 유지. joint hazard 재보정은 P90≤P150 위반 0.
  - A0-3 민감도: s 범주형 ±0.000 / 사건 가중 −0.028 / 수술당 단일
    landmark **−0.104**(반복 landmark 구조가 성능의 실질 일부) /
    fit-side full-refit bootstrap CI **[0.597, 0.702] / [0.600, 0.700]**
    (mean 0.668/0.664) — "실질 대역 ~0.60-0.70" 표기 재확인.
- **A1 완료** — `results/phase3/A1_BULLPEN.md` + gs_flags_v1.parquet:
  - GS 빌드 sanity 통과(연 4,860, 2020 1,796). 3분류: test RP 7,888 /
    SP 3,646 / swing 551; opener·bulk 전담 0건(서술 5분류).
  - **모델 변형 전부 널** (H150 primary: pooled −0.001 / interaction
    −0.010 / 분리 +0.000 / Cohen relx-drift −0.008, 전부 CI 0 포함) →
    feature/모델 경로 종료. Cohen은 사전 등록대로 널 즉시 종료.
  - **quota 채택**: 안정 영역 q\*=20 → mature test에서 RP 사건 포착
    **0→5** (양 H), 총 recall 손실 정확히 2건(21→19, 24→22) — 게이트
    (RP↑ & 손실 ≤2) 통과. 운영 정책 "top-50 중 RP 상위 20석 예약"을
    동결 대상에 포함. 정책 Pareto로만 인용(모델 ROC 개선 아님).
- **A-IL 완료 — canonical 미채택 (blackout 게이트 실패)** —
  `results/phase3/AIL_RESULTS.md`:
  - 수집·파싱 게이트 통과: episode 6,692 (팔꿈치 1,033), 연도별 sanity
    정상, 무작위 40건 수동 검수 정밀도 40/40 (한계: "flexor" 미포함 —
    동결 셋 유지). 정보시점 = transaction date, episode 병합 적용.
  - 전체 feature는 dROC +0.068/+0.046 EXCL0 (recall@50 24→44)로 극적
    개선이지만, **blackout 60d에서 −0.006/−0.003 (승격 기준 실패)**,
    lead 중앙값 34일(P25 13일), 이력 없는 포착 사건 0% — 신호의 실체는
    공개 진단의 메아리. **사전 등록대로 "공개 진단 후 triage 신호"로
    분류, canonical 제외.** 조기경보 서사는 경기 데이터 모델(B'의
    lead 72일)에만 남는다.
- **동결 완료** — `results/phase3/FROZEN_MODEL.md`: M_sa+hazard 계수·
  scaler·재보정(a −0.164, b 0.966) 고정, 경보 정책(top-50, RP 예약
  20석, 순위 P150) 포함. **2025 점수 timestamp 아카이브**
  `frozen_scores_2025_ts20260713.csv` (3,994 창, md5 a3304...) — 라벨
  성숙 시 이 파일로만 robustness 평가(재채점 금지). 이후 MLB 모델 변경
  금지; 남은 트랙 = KBO 이전(계수 재적합) + 제안서(사용자) + triage
  별도 계층.

### 2026-07-13 (계속 4) — 2026 다운로드 승인, 진짜 전향 채점 시작, 회고 문서

- **사용자 승인으로 2026 Statcast 부분 시즌 다운로드** (3/1-7/12 스냅샷,
  459,341구; src/download_statcast.py에 END_OVERRIDE 추가).
- **진짜 전향 채점 1회차 완료**: 결정일 2026-04/05/06/07, 2,654 창,
  동결 계수·재보정 재현 assert 통과 후 채점(재적합 없음) →
  `results/phase3/prospective_scores_2026_ts20260713.csv`
  (md5 1f7648ef...). 8/9월 채점은 월 데이터 갱신 후 동일 스크립트
  (`score_2026_prospective.py`)로 grid만 확장.
- **회고 문서 작성**: `PROJECT_RETROSPECTIVE.md` (사용자 시점 —
  숫자의 여정, 통한 것 9가지, 안 통한 것 6가지, KBO로 가져갈 것).
- 남은 것: 제안서(사용자, 재료 완비) / KBO 이전 패키지 / 2026-08·09
  채점(데이터 갱신 시) / 2025 robustness 평가(~2027 중반).

### 2026-07-13 (계속 5) — codex 3차 감사 수용 (fable 재계산 검증 후 전면 반영)

- **검증**: ① quota safety 재계산 — codex 표와 완전 일치 (safety H150
  q=0 16 → q=20 12, 손실 4 > 허용 2; q=5는 전 셀 +1이지만 사후 관찰).
  ② A-IL parser 소급 — 정확히 33/1,033 episode (최대 773일, 중앙값
  33일); 공개일 기준 blackout 60d dROC = fable −0.002/+0.001 (codex
  −0.0005/+0.0009와 구현 차이 내 동일 결론 ≈0). ③ 코드 검토로 CITL
  명칭·window slope·one-landmark 음성행·2026 시점 논리·score 스크립트
  재적합 방식 전부 확인. **6개 권고 전부 수용.**
- **정정 반영**:
  1. **q=20 채택 철회 → 조건부 challenger 강등** (canonical 경보 =
     q=0 top-50). 기존 점수 파일의 alert_q20 열은 challenger 기록으로
     보존, q=5 소급 채택 금지. → A1_BULLPEN.md 정정 헤더 +
     FROZEN_MODEL.md 4절.
  2. **2026년 4-7월 채점 재분류 = label-blind delayed shadow backfill**
     (결정일이 채점일보다 과거 + 2026 라벨 일부가 시트에 존재). **첫
     진짜 전향 cohort = 2026-08-01을 당일 이전 채점·해시·저장할 때.**
  3. A0 표현 정정: "체계적 낙관 없음"·수치적 "하한" 삭제(3 fold 변동
     ±0.07), citl = mean-pred − prevalence로 명칭 정정, 구간 항등 ≠
     window 보정(slope 0.24/0.55), slope 1.9 원인 단정 불가,
     one_landmark = outcome-dependent stress test. → A0_RESULTS.md.
  4. A-IL parser 한계 명시 + 공개일 기준 정정 blackout ≈0 (**미채택
     결론 불변**), "triage 후보(재검정 필요)"로 강등. → AIL_RESULTS.md.
  5. "~0.60-0.70" = sensitivity envelope (공식 CI 아님);
     "KBO 기대 0.56-0.68" → **MLB novel-pitcher stress reference**
     (KBO 성능 예측 아님). → FROZEN_MODEL.md 5절, 회고 문서.
  6. **동결 무결성 보강**: `frozen_model_state.json` (full-precision
     정본, SHA-256 e14ba800...) 생성; 점수 파일 SHA-256 기록 (2025
     ca7c4e73..., 2026 ab7e8b87...); 향후 채점은 상태 로드 방식 +
     append-only + 사용자 git commit/tag 권고.
- 회고 문서도 동일 정정 반영 (quota·감사 3회·envelope·stress reference).
- **사용자 결정 (2026-07-13)**: 불펜 별도 정렬 보조 명단은 **KBO 적용
  후 검토 항목으로 보류** — PROJECT_MEMORY 4절 열린 후보 +
  KBO_applicability.md 5b절에 기록.
- **데모 시트 생성 (사용자 지시)**: 테스트 기간 월별 top-20 명단
  (이름·P90/P150·근거 분해·실제 수술 여부) →
  `results/phase3/demo_test_top20.csv` + `scripts/demo_test_top20.py`
  (동결 계수 재현 assert, 읽기 전용 — canonical 불변, 데모/제안서용).
  적중 예시: Mize(6/1 1위 → 14일 후 수술), 류현진(5/1 5위 → 48일 후),
  Buehler(8/1 2위 → 22일 후). top-20 행 360개 중 90일 내 수술 19행
  (11명). **KBO 이전 후 로컬 웹 대시보드(순위·근거·읽는 법·검색)
  구축 아이디어 — 사용자 제안, KBO 단계 검토 항목** (KBO_applicability
  5b절에 추가).
- **문서 재정리 (사용자 지시)**: PROJECT_RETROSPECTIVE.md **삭제**,
  대체 = **`progress_MLB.md`** — 7/8 이후 진행을 일반 독자용으로 재작성
  (사용자가 7/7까지는 직접 정리함). 내용: 입력/출력 정의(datapoint =
  투수×결정일, t 이전 정보만), feature 9개 표, 로지스틱+5칸 구조의
  정확한 의미(공통 감쇠, 순위 동치), 표준화는 학습 구간 자 고정, 절대
  확률 인용 규칙, 계수 해석·위험 프로필, 감사 3회·인용 규칙, 기각
  목록, 불펜 상태, 데모 사례, 동결·남은 일 — 사용자 질문에서 드러난
  불명확 지점들을 정면으로 다룸.

### 2026-07-13 (계속 6) — KBO 이전 계획 제안 v1 (승인 대기, 미실행)

**전제 (계획의 지렛대)**: 동결 모델의 입력 = 경기별 투구수·등판일·선발
여부 (+구속 추세, 기여 ≈0). 트래킹 불필요(MLB에서 증분 널). → KBO
공개 박스스코어 수준으로 feature 8/9개 계산 가능성 높음. 진짜 병목은
**부상 라벨** (공개 TJS 시트 부재; 뉴스 재구성은 1군 편향, 연 5-15건
추정 — R4). 라벨은 TJS-only 유지(팀원 트랙 충돌 방지).

**단계 (각 단계 착수는 별도 "시작" 지시)**:
- **K0 — 공개 소스 정찰 (읽기 전용, 1-2일)**: ① KBO 경기별 투구수·
  선발 여부·등판일의 공개 가용성 확정 (KBO 기록실/스탯티즈; 연도
  커버리지, 수집 방식·약관 확인) ② 구속 가용성 (없어도 진행 가능
  명시) ③ 뉴스 기반 TJS 라벨 v0 스코핑 (검색 전략·예상 건수·수술일
  정밀도). **게이트: 투구수 ≥6시즌 + 라벨 ≥30건 전망** — 미달 시 계획
  수정 후 재보고.
- **K1 — 공개 데이터 구축**: 경기 로그 수집 + sanity(연도별 경기수·
  투구수 분포), 뉴스 라벨 시트 v1(수술일 정밀도·하한 명시), cohort
  이식 (결정일 grid는 KBO 시즌에 맞춰 사전 결정 — 기본 제안 4-9월
  유지), feature 8개(+구속 가용 시 9개), 표준화는 KBO 학습 구간 기준.
- **K2 — 예비 재적합·정직 평가**: 시간 분할 fold(가용 연도에 따라
  사전 등록), 투수-clustered bootstrap, rolling 방향 검사. **성공 기준
  (사전 등록) = 파이프라인 작동 + 방향성 확인 — 사건 수 부족으로 CI가
  넓을 것을 명시, 내부 라벨 확보 전 "성능 주장" 금지.** 선택: MLB 동결
  계수 zero-shot vs KBO 재적합 비교 리포트 (계수 이식 아님 — 참조
  실험).
- **K3 — 제안 패키지**: 제안서(사용자) + progress_MLB.md + KBO 예비
  결과/데모 (가능하면 대시보드 프로토타입). **데이터 요청 우선순위
  재정의: ① 부상자명단 상세(부위·일자·진단, 2020~) ② TJS/UCLR
  수술일 ③ (2차) 트랙맨 원자료 — MLB에서 트래킹 증분 널이었음을
  정직 명시, "검증 목적" 프레임.** 프라이버시 친화 검증 제안 카드:
  우리 점수를 사전 봉인 → 구단이 내부 라벨로 자체 채점 (라벨 반출
  불필요).
- **K4 — 협조 후**: 내부 라벨 본평가 → 재적합·재보정 → quota·불펜
  보조 명단 재검토 (KBO_applicability 5b) → 월별 명단 + 대시보드 배포.

**사용자 결정 필요**: ① K0 착수 여부 ② 라벨 수집 방식 (수작업 vs
반자동 — 약관 확인 선행) ③ 결정일 grid (4-9월 유지 vs KBO 시즌 맞춤)
④ 팀원 트랙과의 조율 확인. **미실행 상태 — "시작" 지시 대기.**

### 2026-07-13 (계속 7) — KBO 이전 계획 v2 (codex 감사 정정, 승인 대기·미실행)

**판정**: K0-first, 라벨 우선, MLB 계수·quota 비이식이라는 v1의 큰
방향은 정당하다. 다만 공개 뉴스 라벨은 미보도자를 음성으로 확정할 수
없는 **positive-unlabeled(PU) 확정 사례 registry**이므로, 이를 0/1
라벨처럼 사용한 K2 재적합·ROC·방향성 판정은 철회한다. v1과 충돌하면
아래 분기형 v2가 우선한다. 각 단계는 여전히 사용자의 별도 `시작` 지시
후에만 착수하며, 이 절 작성 시 코드·수집·실험은 실행하지 않았다.

**선택지와 권고**:

| 안 | 공개 단계 | 내부 라벨 전 성능 주장 | 판정 |
|---|---|---|---|
| A — v1 유지 | 뉴스 라벨로 예비 재적합 | 방향성·예비 ROC | 기각(PU 편향) |
| **B — 분기형 v2** | exposure 파이프라인 + 확정 사례 기술분석 | **금지** | **권고** |
| C — 전면 대기 | 내부 라벨 전 아무것도 구축하지 않음 | 없음 | 과도하게 보수적 |

**고정 feature mapping (결과를 보기 전에 데이터 QC로 경로 결정)**:
- **KBO-B7 (public boxscore anchor)** = pc_chronic, pc_acute_dev,
  days_since_last, month, start_share, prior_pc_rate, ncg_log. 구속이 전혀
  없으면 vel_trend와 vt_missing은 모두 식별 불가능한 상수이므로
  "8/9개"가 아니라 **7개**다.
- **KBO-V9** = B7 + vel_trend + vt_missing. 일자별 구속의 정의·단위·
  장기 coverage·측정체계 정합이 gate를 통과할 때만 별도 secondary로
  연다. MLB에서 Tier-2 증분을 검출하지 못했다는 결과는 tracking이
  KBO에서도 무가치하다는 뜻이 아니라, B7의 선행 조건이 아니라는 뜻이다.
- frozen start_share는 실제 GS가 아니라 trailing 365일 등판 중 **50구
  이상 경기 비율**이다. 공식 GS share는 역할 평가·불펜 정책용 별도
  변수이며 anchor feature를 몰래 대체하지 않는다.
- 구현 정본의 vel_trend 기준은 "직전 시즌"이 아니라 현재 연도보다 앞선
  **모든 시즌의 투구수 가중 구속**(없으면 t-30일 이전 이력 fallback)이다.
  KBO 구현 전 코드 기준 feature dictionary와 단위·censoring crosswalk를
  먼저 봉인한다.

**K0 — read-only feasibility + 권리·접근 gate (수집 없음)**:
1. 공식 KBO/KBOP 또는 허가된 공급자에서 player/game ID, 경기일,
   등판별 투구수, 정정·더블헤더·서스펜디드 경기, 팀 이동을 연결할 수
   있는지 수동 schema 표본으로 확인한다. 최소 6개 완결 decision season에
   더해 365일 burn-in·직전 시즌·as-of 경력 경기수를 계산할 pre-history가
   필요하다.
2. **페이지 공개와 대량 수집 허가는 별개**다. KBO/STATIZ 약관, 저장·
   파생물 공개 범위, 재현 snapshot 허용 여부를 확인하고, export/API 또는
   서면 사용 허가가 없으면 K1 자동수집을 시작하지 않는다. STATIZ
   scraper·비공식 API는 서면 승인 전 제외한다.
3. 뉴스/구단발표에서 실제 시행된 UCLR/TJS를 찾는 검색 universe와 증거
   ontology를 고정한다: A=시행+정확일, B=시행 확인+날짜 구간,
   C=예정/발표일·술식(UCLR vs repair/internal brace) 불명확. 발표일,
   예정일, 시행일, primary/revision/repair, 출처를 별도 필드로 둔다.
4. 기존 `투구수 ≥6시즌 + 라벨 ≥30건 전망`은 폐기한다. 30건은 case
   discovery 목표일 수는 있어도 supervised gate가 아니다. 실제 확인한
   distinct 사건, 날짜 정밀도, eligible universe의 ascertainment와
   train/valid/test별 사건 수·예상 CI 정밀도를 보고한다.
5. 결정일은 outcome을 연결하기 전에 봉인한다. **비교 anchor는 4/1–9/1
   고정**, season-aware 30일 grid는 운영 challenger로만 사전 정의하고,
   2020 지연 개막은 별도 민감도로 둔다. K0 schedule matrix를 본 뒤 둘의
   최종 지위를 정하되 TJS 결과는 보지 않는다.

**K1-F — 승인된 exposure·feature/cohort 구축**:
- 허가된 소스만 사용해 1군 정규시즌 등판을 primary로 구축하고, raw
  snapshot·source/version/hash·ID crosswalk를 보존한다. t 당일 경기는
  제외한다. primary risk set은 결정일 현재 **KBO 구단 통제 roster**
  (1군·2군·IL 포함) 중 ≥20 KBO 1군 경기·최근 365일 내 1군 등판을
  충족한 투수로 정의한다. 공개 roster history가 이를 완전히 복원하지
  못하면 approximate cohort로 표시한다. 해외/은퇴/방출 뒤 horizon의
  수술 follow-up이 불완전한 창은 음성이 아니라 censor하며, formal
  binary/time-dependent metric 처리 규칙을 결과 조회 전에 고정한다.
- B7을 primary anchor로 고정한다. V9는 K0 velocity gate 통과 시에만
  동일 coverage의 secondary로 구성한다. 2군·재활·불펜 세션 등 숨은
  workload는 공개 B7의 결측/오분류 한계로 기록하고 내부 요청 항목으로
  넘긴다.

**K1-L — 공개 확정 사례 registry (음성 라벨 생성 금지)**:
- A만 exact-date primary case, B는 interval-censor/날짜 민감도, C는
  discovery 전용으로 둔다. 비보도 선수는 0이 아니라 **unknown**이다.
- 공개 확인 사건 수는 사건 수의 하한일 수 있지만, 선택편향 때문에
  ROC/성능의 하한은 아니다. loss-to-follow-up과 KBO 이탈 후 수술도
  음성으로 강제하지 않는다.

**K2-public — engineering/domain-shift 기술 감사만**:
- 허용: source·ID·coverage·as-of/leakage QC, B7/V9 support·결측·분포,
  frozen MLB state를 그대로 쓴 zero-shot 순위의 기계적 transport 점검,
  확정 A/B 사례의 당시 percentile·lead를 **편향된 case-series**로 보고.
- 금지: 뉴스 unknown을 음성으로 둔 KBO 재적합, ROC/PR/Brier/lift,
  prevalence·calibration·절대확률, feature/정책 선택. null을 포함해 모든
  사전 endpoint를 보고하며 "방향성 확인"을 성공 기준으로 삼지 않는다.
- zero-shot은 `frozen_model_state.json`의 **MLB scaler+계수**를 함께 쓰고
  순위 transport stress로만 부른다. 구속이 없을 때는 `vel_trend=0,
  vt_missing=1`인 **missing-velocity transport stress**로 명명한다.
  KBO scaler를 쓰면 zero-shot이 아니다.

**K3 — 데이터 협력·거버넌스 제안**:
1. 최우선: eligible roster 전체에 대한 실제 UCLR/TJS 시행 여부·정확일·
   primary/revision/repair 구분·stable pseudonymous ID·follow-up 완전성.
   KBO가 exact surgery registry를 보유한다고 단정하지 않고 존재·형태부터
   확인한다.
2. 1·2군/재활 등판 workload, 로스터·이적·실제 GS/역할, 가능하면 숨은
   투구량. 공개 source 권리가 불명확하면 승인된 export/API도 함께 요청한다.
3. IL 부위·진단·일자는 case ascertainment/auxiliary용이며 TJS primary
   label을 대체하지 않는다. tracking 원자료는 B7 이후 증분 검정용 4순위다.
4. 라벨 반출이 어렵다면 점수/컨테이너를 기관 내부에서 실행하고 aggregate
   metric만 반환한다. 데이터사용계약·기관/의료/개인정보 검토, 최소권한,
   가명화와 audit trail을 제안서에 포함한다.

**K4 — 내부 완전 라벨 확보 후 supervised refit·locked 평가**:
- reliable label end와 H90/H150 censoring을 고정한 뒤, 가장 최근의 완결
  2시즌=test, 직전 1시즌=valid, 그 이전=train인 forward-only 원칙을
  우선한다. 실제 distinct 사건 수와 사전 정한 CI/power 정밀도가 부족하면
  fold를 결과에 맞춰 합치지 않고 formal 평가를 보류한다.
- KBO refit만 KBO train scaler를 사용한다. 고정 비교군은 **KBO-B4**
  (pc_chronic, pc_acute_dev, days_since_last, month)와 **KBO-B7**이며,
  둘 다 MLB와 같은 discrete-time hazard LR(L2, C=1, class_weight 없음,
  s=0..4)를 첫 anchor로 쓴다. V9·prior_tjs 등 challenger는 각 데이터
  gate 통과 후 valid에서만 선택해 untouched temporal test에 1회 적용한다.
  penalty/feature/model grid를 추가하려면 결과를 보기 전 별도 사양으로
  봉인한다.
- **primary endpoint = 용량 정규화 budget에서 distinct 수술을 한 번만
  세는 event-level recall**. pooled-window ROC/PR은 secondary이며,
  event-weighted 양성 민감도와 outcome-independent 단일 관측(예: 투수-
  시즌 첫 eligible decision) 민감도를 의무 보고한다. 투수-clustered
  paired bootstrap·forward rolling을 유지한다.
- formal 실행 전 label-blind event-count/power audit로 최소 fit/test 사건,
  허용 CI 폭과 paired 채택 규칙을 먼저 문서화한다. 원칙은 paired CI가
  0을 배제하거나 사전 고정 rolling 방향 규칙을 만족하는 경우뿐이며,
  사건 부족 시 결과에 맞춰 기준을 낮추지 않는다.
- calibration은 complete labels의 train/valid에서 **window P(H)** 기준으로
  적합하고 test에서 intercept/slope/Brier를 확인한다. 공개 PU 단계에서는
  하지 않는다.
- MLB top-50·q20은 이식하지 않는다. 월 적격 투수의 top 5/10% 또는
  구단당 n명처럼 운영 용량 정규화 budget curve를 보고, RP quota/별도
  명단은 valid에서 1회 선택해 test에 고정한다.

**K5/K6 — silent prospective → guarded internal pilot**:
- K5: 결정일 전 cohort·state·score를 append-only로 해시·봉인하고 H90/
  H150 성숙 전 변경하지 않는다. 첫 score 봉인 전에 최소 distinct 사건,
  primary event-recall budget, 허용 CI, calibration·drift 기준을 포함한
  go/no-go를 문서화하고, 통과 전에는 성능·배포 주장을 하지 않는다.
- K6: 통과 시에만 제한된 내부 pilot. 의료·workload review의 보조 신호로
  사용하고 진단·계약·방출·로스터 결정을 점수 하나로 내리지 않는다.
  공개 실명 위험 명단은 만들지 않으며 접근통제·drift/calibration·오경보
  모니터링을 포함한다. 대시보드는 이 단계 이후의 인터페이스다.

**v2 사용자 결정 포인트**: ① K0의 공식 export/API·연구 이용 문의 허용
여부 ② K0 이후 fixed 4–9 anchor와 season-aware challenger의 최종 지위
③ 공개 case registry의 수동/허가된 반자동 방식 ④ 내부 협조 대상(KBO/
KBOP vs 개별 구단) ⑤ 팀원 트랙과의 ID·산출물 경계. 현재 전부 미실행이다.

### 2026-07-13 (계속 8) — 실행 전 중복 감사 + 연구 대시보드 배포 추가 (미실행)

**명칭 정정**: 앞서 codex가 제안한 준비 단계 `R0–R2`는 저장소의 기존
Phase 3 블록 `R1_ROLLING`·`R2_AGE`와 이름이 충돌했다. 기존 R1 rolling과
R2 age는 이미 완료된 결과다. 아래 준비 작업은 **M0–M2**로 개명하며,
Claude가 완료한 부분을 다시 실행하지 않는다.

| 준비 단계 | 읽기 전용 감사 판정 | 남은 범위 |
|---|---|---|
| **M0 — 동결 기준점 봉인** | **완료** | `frozen_model_state.json` full-precision state, 2025/2026 score archive와 SHA-256가 commit `7e1f72e`에 포함됨. 입력 snapshot까지 묶은 단일 manifest는 선택 보강일 뿐 선행 작업이 아님. |
| **M1 — 감사 지적 구현 보수** | **미완료** | A0는 문서 정정만 완료; 표준 calibration intercept/grouped-OOF final P(H)/outcome-independent landmark 산출물이 없음. A1은 q20 철회 결론만 정정; GS 중복 판별, safety 행·게이트, paired dPR 저장이 남음. A-IL은 문서 정정만 완료; 공개일·episode closure·post-TJS parser와 alert-time history/lead 보수 및 corrected CSV 재생성이 남음. |
| **M2 — state-load 채점 경로** | **부분 완료** | state dump는 완료. 실제 scorer는 아직 train/TJS label을 읽어 scaler·LR·재보정을 다시 fit하고 파일을 덮어쓸 수 있음. 검증 loader, fit-free scorer, q0 출력, append-only writer/manifest, golden·leakage·overwrite 테스트가 남음. |

M1은 **이미 동결한 MLB 성능을 다시 선택하거나 개선하는 실험이 아니라**,
감사 결론을 재현 코드·CSV·본문까지 일치시키는 정합성 보수다. M2도 계수
변경 없이 동결 state를 운영 경로에서 직접 사용하는 구현 보수다. 기존
2025/2026 archive는 이력 보존을 위해 수정하지 않는다.

**실행 순서와 ETA (사용자 `시작` 전 미실행)**:

1. **M1 감사 보수·재검증 — 1–2일**: A0/A1/A-IL을 독립 패치하고 각
   corrected artifact와 결정 메모를 생성한다. canonical q=0·TJS-only와
   동결 계수는 변경하지 않는다.
2. **M2 load-only scorer hardening — 0.5–1일**: state schema/hash 검증,
   `fit()`·학습 라벨 의존이 없는 scorer, 결정일별 exclusive-create,
   append-only manifest와 회귀 테스트를 완성한다.
3. **D0 제한형 연구 대시보드 구현·배포 — 1–2일**: M1/M2 통과 후
   MLB frozen archive와 retrospective demo를 입력으로 사용한다. 공식
   제안서나 KBO 협조를 기다리지 않는 별도 연구 산출물이다.
4. 이후 **K0 → K1-F/K1-L → K2-public**은 계획 v2대로 진행한다. 공개
   데이터 단계에서는 성능 주장 없이 engineering/domain-shift만 감사한다.
   complete internal label 이후의 K4–K6 운영 경로도 기존 게이트를 유지한다.

**D0 대시보드 고정 범위**:

- 결정일별 q=0 순위, 투수 검색, P90/P150, 표준화 feature 기여 분해,
  역할 필터와 점수 읽는 법, model/data hash·snapshot 시점을 표시한다.
- 대시보드는 versioned CSV/manifest를 **읽기만** 하며 모델 fitting·재보정·
  데이터 다운로드를 하지 않는다. retrospective outcome은 명시적으로
  분리된 demo view에서만 보인다.
- 배포 산출물은 우선 **로컬 one-command + 접근 제한형 연구 인스턴스**다.
  공개 실명 실시간 위험 명단은 만들지 않고, 절대확률 한계·sensitivity
  envelope·불펜 coverage 한계를 화면에 고정한다. 외부 호스팅 계정·비용·
  credential이 필요하면 실제 배포 직전에 사용자 승인을 받는다.
- 수용 기준: frozen golden score/rank와 일치, feature 순서 오류 차단,
  `game_date < t`, q=0 정책, 기존 snapshot 덮어쓰기 거부, 새 데이터 없이
  재실행 시 동일 화면을 자동 검증한다.

**KBO 연동 분리**: K1-F/K2-public 이후에는 익명·집계 중심의 engineering
adapter만 D0에 연결할 수 있다. 투수 실명 운영 dashboard는 complete label로
K4 locked 평가와 K5 silent prospective의 사전 go/no-go를 통과한 뒤 K6
제한형 pilot에서만 연다. 따라서 **연구 대시보드 배포는 이번 연구 범위에
포함**하되, 검증 전 KBO 위험 운영 배포로 해석하지 않는다.

이 절 작성 시 프로젝트 스크립트·실험·다운로드는 실행하지 않았고 코드·
데이터·설정도 수정하지 않았다. 실행은 사용자의 다음 `시작` 지시를 기다린다.

### 2026-07-14 — M1/M2 실행·교차감사, K0 판정, D0 구현 완료

사용자가 `시작`을 명시해 "계속 8"의 M1→M2→D0와 계획 v2의 read-only
K0를 실행했다. 기존 MLB 동결 모델·계수·q=0 정책·TJS-only 라벨은
변경하지 않았고, 기존 archive/legacy CSV도 덮어쓰지 않았다.

**환경·독립 감사**

- `.venv` 환경 검증은 정상 권한에서 **9/9 PASS**. 최초 sandbox 실행의
  pybaseball cache 쓰기 1건 실패는 권한 격리 때문이었고, 승인된 동일
  환경에서 실제 단일일 fetch까지 통과했다.
- M1 세 블록은 각 구현자와 별도의 교차감사자가 재실행·대조했다. corrected
  CSV/parquet는 저장본과 byte-identical했고, 시간 경계·selection·paired
  CI·문서 수치를 독립 확인했다.
- 교차감사에서 발견한 정합성 3건도 반영했다: opener 기술 기준 `<40`→
  사전등록 `<30`, `A1_separate` 명칭을 실제 구현인 RP/non-RP로 정정,
  corrected A-IL의 비채택 근거를 "blackout 방향 실패"가 아닌
  result-informed parser 보수로 confirmatory 지위 상실+CI 0 포함으로 정정.

**M1-A0 — PASS, canonical 불변**

- `a0_recal.py`, `a0_sens.py`를 보수하고 legacy 산출물과 분리해
  `a0_recal_corrected.csv`, `a0_recal_oof_window.csv`,
  `a0_sens_corrected.csv`를 생성했다.
- 투수-grouped 5-fold OOF 최종 P(H): n=16,949, 투수 1,029명,
  양성 window 173/252. mean error −0.000170/−0.000161, 표준 fixed-slope
  calibration intercept +0.0170/+0.0111이나 joint slope 0.2658/0.5683.
  평균 일치나 interval 재보정만으로 window calibration을 주장할 수 없다.
- mature test raw 표준 절편 +0.363/+0.308, joint slope 1.940/1.962,
  ROC 0.7009/0.6958. 절대확률 인용 금지는 유지한다.
- outcome-blind `(투수, 달력연도)` 첫 적격 decision 규칙은 fit-only
  0.6225/0.6250, 같은 first-landmark 평가집합에서 0.5642/0.5850
  (canonical fit 0.6495/0.6435). 역사적 −0.104는 계속
  outcome-dependent stress로만 남긴다.

**M1-A1 — PASS, q=0 유지**

- actual first-pitch GS v2: 2016–2025 정규시즌 **22,764/22,764경기 모두
  정확히 2 GS**, 예외 0. legacy v1의 mid-at-bat 거짓 GS 6행은 별도
  diff로 보존했다.
- q20은 primary H90/H150과 safety H90 local gate는 통과하지만 safety
  H150에서 **16→12, RP 0→2, 총 손실 4**로 실패한다. 최종
  `adopt=False`, `canonical_q=0`; q5 전 셀 +1은 사후 탐색이다.
- separate paired dPR 네 셀 모두 저장했고 CI가 모두 0 포함한다. 실제
  구현은 RP vs non-RP이며 사전등록 shorthand "SP/RP"와의 편차를
  명시했다. descriptive opener/bulk는 등록 기준 GS당 `<30구`에서도 0건.
- 정본: `a1_bullpen_corrected.csv`, `gs_flags_v2.parquet`, GS sanity/
  exception/v1-v2 diff CSV. 기존 v1과 `a1_bullpen.csv`는 보존했다.

**M1 A-IL — 계산 PASS, canonical 미채택 유지(근거 정정)**

- 공개 disclosure date를 보존하고 placement/transfer/activation 모두에
  60일 gap을 일관 적용했다. 13,827 actions→6,816 episodes; activation
  4,101, gap-timeout 2,031, right-censored 684. 완료 procedure ontology와
  미래 의도·비팔꿈치 UCL false-positive 회귀검사를 통과했다.
- corrected 전체 dROC는 +0.06862/+0.04453, blackout 60일은
  **+0.00577/+0.00232**이나 CI [−0.02787,+0.04803]/
  [−0.02067,+0.02990]로 모두 0 포함. 첫 caught alert→수술 lead 중앙값은
  36/37일이다.
- 중요한 판정 정정: 위 60일 점추정은 과거 방향-only 조건을 기계적으로
  통과하므로 "blackout gate 실패"라고 부르지 않는다. 그러나 parser
  정의는 결과를 확인한 뒤 감사에서 보수돼 confirmatory 지위를 잃었고,
  독립적 효과의 CI도 0 포함한다. 따라서 **소급 채택하지 않고 새 고정
  parser/독립 자료에서 재검정할 triage 후보**로 유지한다.
- 정본: `il_episodes_asof_v2.parquet`, `ail_results_asof_v2.csv`,
  `ail_alert_events_asof_v2.csv`. legacy 파일은 보존했다.

**M2 — PASS, fit-free frozen scorer 완성**

- `frozen_state.py` loader가 canonical SHA-256 `e14ba800...`, schema,
  feature 순서, 벡터·유한값·scale을 검증한다. NumPy equation만 사용하며
  미래 CLI는 학습 cohort/TJS label/scikit-learn/`fit()`에 의존하지 않는다.
- strict `game_date < t`, actual first-pitch GS, q=0 top-50, raw/recal
  P90/P150, feature contribution을 출력한다. 순서 좌표 결측·exact tie·
  경기당 starter≠2는 score 전에 실패한다.
- score CSV+manifest는 exclusive-create이며 state/input/output SHA와
  as-of metadata를 함께 저장한다. 기존 archive는 변경하지 않았다.
- 테스트 **9/9 PASS**. 2026-04~07 delayed-shadow 2,654/2,654행에서 네
  확률열 atol 5e-8, rank 전 행 일치; feature 순서 drift, 당일 경기 누출,
  q0, overwrite, first-pitch tie를 각각 실패/차단했다.

**D0 — 로컬 production 산출물 PASS, 원격 publish만 대기**

- `dashboard/`에 retrospective demo 전용 vinext Sites 앱을 구현했다.
  q=0 순위·P90/P150·기여 분해·검색·역할·결과 분리·model/data hash와
  절대확률/비임상 경고를 제공한다. 현재 실명 전향 위험 명단, fitting,
  재보정, 다운로드 기능은 없다.
- `demo_test_top20.csv`는 원본과 byte-identical(SHA-256 `d2f68cd3...`),
  manifest에 frozen state와 asset hash를 고정했다. production build,
  lint, server-render/data identity 테스트 **2/2 PASS**. patched PostCSS로
  `npm audit --omit=dev` production 취약점 **0건**; 개발 도구 audit 9건은
  런타임 비포함이며 강제 major fix를 적용하지 않았다.
- 외부 Sites publish는 아직 수행하지 않았다. Sites는 pushed source의
  commit SHA를 요구하지만 프로젝트 규칙상 git commit/push는 사용자가
  직접 한다. source commit/push 뒤 owner-only 접근으로 저장·배포하는
  마지막 단계만 남는다. 빈 site를 미리 만들지는 않았다.

**K0 — `PARTIAL`; K1 자동수집 `NO-GO`**

- 공식 KBO 퓨처스 실제 박스스코어의 날짜·선발/구원·투구수와 URL의
  gameId, 선수 페이지의 playerId/등판 역할을 수동 확인했다. B7 원자료의
  기술 가능성은 있으나 1군 다년 coverage·ID 계약·historical risk-set
  replay는 미확인이다.
- 2025 TrackMan 공식 구속 일원화는 측정체계 사실만 PASS. 공개 structured
  daily velocity가 없어 V9 공개 경로는 FAIL이다.
- 공식 export/API 또는 연구 bulk 저장·파생물 재배포 허가를 확인하지
  못했고 약관은 사전승낙 범위를 둔다. 공개 열람은 자동수집 권리가
  아니므로 **K1-F scraper/비공식 API/대량 다운로드를 시작하지 않았다.**
- 공개 UCLR/TJS 기사는 A/B/C evidence registry 후보일 뿐 PU다.
  `unknown != negative`; 공개 자료만으로 refit·ROC/PR·calibration은 금지.
- 다음 KBO 순서는 권리 확보→소수 경기 schema/completeness pilot→B7-only
  여부 재심사다. 상세 정본은 `results/kbo/K0_FEASIBILITY.md`.

**남은 승인·시점 작업**

1. 사용자가 dashboard source를 commit/push한 뒤 D0 owner-only Sites
   production publish를 완료한다.
2. KBO 공식 export/API/서면 허가 없이는 K1을 열지 않는다. 허가가 생기면
   전기간 수집이 아니라 소규모 schema/completeness pilot부터 별도 승인한다.
3. 2026-08-01 첫 진짜 전향 cohort는 **8/1 당일 이전** 최신 raw snapshot으로
   M2 CLI를 실행·해시·봉인한다. 현재 7/12 snapshot으로 미리 만들지 않았다.

### 2026-07-14 (계속 2) — KBO 후속 방향·팀 인계 정비

K0 이후 사용자가 과거 KBO 홈페이지 수집 경험과 네이버 문자중계의 PTS
구속 경로를 제시했다. 이에 따라 기존 K0 문서의 “공개 원자료가 전혀 없다”는
식의 해석은 사용하지 않는다. 다음 실행 후보는 Codex가 직접 수행하는
**label-blind 소규모 source/schema/coverage pilot**이며, B7은 KBO 공식
투구수를 정본으로 하고 구속은 측정계·기간 단절을 먼저 확인하는 별도
`V9-PTS` 후보로 둔다. 공개 기사 TJS는 계속 A/B/C 확정 사례 registry이며
미발견을 음성으로 바꾼 학습·ROC/PR·보정은 금지한다. 이 절에서는 KBO
수집이나 실험을 시작하지 않았다.

팀원이 GitHub만 보고도 연구 흐름과 정본을 이해할 수 있도록 다음 저장소
인계 정비를 완료했다.

- 루트 `README.md` 신설: 현재 상태, 정직한 수치 해석, 읽기 순서, 연구
  흐름, 저장소 지도, 환경 구성, 외부 데이터 경계, 작업 원칙을 정리했다.
- 루트 `.gitignore`의 `data/`를 `/data/`로 고쳐 실제 연구 데이터만
  root-anchored ignore한다. 이로써 `dashboard/public/data/`의 byte-locked
  demo CSV와 manifest가 Git 추적 대상이 됐다. `.venv`, upstream clone,
  binary cache, prediction dump는 계속 제외한다.
- README local link **17/17 PASS**. 대시보드 demo CSV는
  `results/phase3/demo_test_top20.csv`와 SHA-256 `d2f68cd3...`로 일치한다.
  production build + server-render/data-identity 테스트 **2/2 PASS**.
- 시점 문구 정정: 앞 절의 “8/1 당일 이전”은 동결 정본과 같은
  **“2026-08-01 당일 또는 이전”**으로 읽는다.
- Drive는 `data/` 전체 1.67GB를 그대로 공유하지 않는다. 기본 current
  canonical 묶음은 v4 cohort/slim/game_features, corrected GS v2, TJS
  snapshot, vdecay(약 29MB)이며 2026 채점 담당 시 partial Statcast 2026을
  추가한다. M2 golden regression 9/9까지 재현할 때는 legacy
  `gs_flags_v1.parquet` 테스트 fixture도 추가한다(합계 약 91.2MB). 전체
  feature 재구축 담당자에게만 raw 2016–2026과
  필요한 A-IL 자료를 확장 제공한다. 저자 `final_df.csv`는 별도 제한
  공유하고 `.venv`, `node_modules`, checkpoints, legacy `windows.npz`,
  logs/pred dumps는 제외한다.

현재 변경은 사용자가 직접 commit/push해야 한다. dashboard의 새 추적 파일
두 개(`demo_test_top20.csv`, `manifest.json`)가 해당 commit에 포함돼야 한다.
