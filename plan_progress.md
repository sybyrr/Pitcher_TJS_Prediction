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
  4. **동결**: 위 완료 후 `results/phase3/FROZEN_MODEL.md`에 스펙+계수
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
