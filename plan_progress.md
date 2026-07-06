# 계획 및 진행상황

목적: 승인된 실행 계획(Phase 0–3)의 canonical 정의와 세션별 진행 로그.
계획 근거는 2026-07-06 논의에서 확정 (B안: 재현 + 진단 + KBO-지향 ablation).
관련 문서: `kang_repo_audit.md`(코드 감사·환경), `reproduction_and_dataset.md`(데이터 정의), `KBO_applicability.md`(KBO 전략).

---

## 계획

### Phase 0 — 데이터 기반 구축
- Statcast 2016–2023 전체 다운로드 (`src/download_statcast.py`, 시즌별 parquet).
- 추출 파이프라인 재작성 (`src/extract.py`): upstream `Pybaseball_extract.py`의 1:1 포팅
  + 문서화된 편차([DEVIATION]/[GAP-FILL] 마커). 라벨은 v1에서 repo 스냅샷
  `list of TJ.csv` 고정 (코호트 드리프트 방지).
- 분류 전처리 포팅 (`src/prep_classification.py`).
- **성공 기준: 코호트 620 (injured 101 / normal 519), diff feature 102, sequence 224.**
  (모델 입력 feature는 new_before_tj_group 포함 103 — 코드 재구성 결과, 논문 표기는 102)

### Phase 1 — 충실 재현 (버그 포함)
- 발견된 upstream 버그(best-weights 미복원, ViT shuffle=False, 회귀 row-level split)를
  그대로 둔 채 10 seed 학습. 학습 시작은 사용자 승인 후.
- **성공 기준: ViT F1 0.73 / ROC-AUC 0.93, 회귀 R² 0.79 (허용 범위 사전 정의: F1 ±0.05).**

### Phase 2 — 진단·교정 재현
- 버그 수정 영향 정량화 (deepcopy 복원, shuffle 수정).
- 회귀 player-level split 교정 → R² 하락 폭 = 논문 수치의 낙관 편향 크기.
- 시간적 외부검증 (2016–2021 학습 → 2022–2023 테스트).

### Phase 3 — 확장 + KBO-지향 ablation
- +2024 확장 (라벨은 Roegele 최신 시트로 갱신). 2025–2026 제외.
- Feature-tier ablation: Tier1(KBO 공개: 구속·구종·회전수) / Tier2(+release pos,
  extension, spin axis) / Tier3(full). Tier1→2 성능 점프 = 구단 설득 자료.
- KBO 제안서 초안.

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
- **잔여**: 회귀 SHAP (Table 10 재현) — 저장된 체크포인트로 재학습 없이
  실행 가능(`train_regression.py`에서 --no-shap 없이, 예상 1~2.5시간).
- 다음: SHAP 실행 → Phase 2 (버그 교정 + player-level split + 시간 검증).
