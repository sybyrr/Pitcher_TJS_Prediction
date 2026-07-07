# Phase 2 수정 후보 — ultracode 검증 결과 및 실행 설계

작성 기준일: 2026-07-06. 근거: 6차원 코드 감사(finder 6) + 발견별 적대적 검증
(증거·영향 2렌즈, 에이전트 82개). **36건 생존 / 2건 기각.** 원 데이터:
워크플로우 wf_8cda58ed 결과(세션 로그). 모든 항목은 G1(재현 보존) /
G2(정직한 평가) / G3(KBO 이전) 기준으로 판정됨.

원칙: **G1 baseline은 동결.** 모든 수정은 별도 변형(variant)으로 학습해
baseline과 대조하고, 결과 CSV는 변형별 경로로 분리한다.

---

## A. 핵심 발견 (HIGH, 실험 변형으로 정량화)

| # | 발견 | 위치 | 요지 | 예상 방향 |
|---|---|---|---|---|
| A1 | 회귀 row-level split | prepforreg.py:81 | 같은 투수의 경기 행들이 train/test에 분산. **검증자 실측: seed 102에서 test 투수 전원이 train에도 존재(novel 0명).** R²=0.78은 "본 적 있는 투수의 보간" 성능 | 수정 시 R² 대폭 하락 (정직한 수치) |
| A2 | best-weights 참조 버그 | ViT.py:142 외 분류 5종 | deepcopy 없이 참조 저장 → 실제로는 최종 epoch 가중치로 평가됨 | 수정 시 소폭 상승 가능 (현 수치가 비관 편향) |
| A3 | fill-artifact 지름길 | Prepfortrain.py:29, prep_classification.py:47 | 0~1310 그리드 강제 + step-fill로 생기는 **상수 꼬리 길이가 클래스와 상관** (부상자가 실관측 시즌이 짧음) → 모델이 부상 역학이 아니라 "평평한 구간 길이"를 학습 가능 | F1 0.73의 일부가 아티팩트일 수 있음 — 기여도 측정 필요 |
| A4 | diff feature의 미래 정보 | extract.py:281 (upstream L507) | 투수별 평균을 **전체 관측 구간**으로 계산해 빼는 구조 — 배치 시점에 재현 불가(미래 경기 포함), 회귀에선 split 누수와 결합 | causal(expanding-window) 기준선으로 재구축 필요 |
| A5 | 회고적 앵커 | extract.py:192 | day 0 = 마지막 경기(사후에만 알 수 있음) → "100일 전 예측"은 전향적 예측이 아니라 **완결된 궤적의 사후 판별** | 전향적 재정의는 별도 트랙(B2) |
| A6 | threshold·지표 설계 | train_classification.py:49,105 | pos_weight=5 학습 + 고정 0.5 threshold = 보정 안 된 임의 작동점. 5:1 불균형에서 F1@0.5는 부적절, KBO(다른 base rate)로 이전 불가 | PR-AUC 주지표화 + valid 기반 threshold 선택 |
| A7 | 통계적 보고 부재 | (설계) | test 부상 20명 → F1 분해능 ~0.05-0.08. 10 seed는 동일 620명의 재분할이라 독립 증거 아님. 순위 중간부는 노이즈 | paired Wilcoxon + 분해능 명시 + 시간적 홀드아웃 |

## B. 코호트·태스크 설계 결함 (MEDIUM — 일부 수정, 일부 문서화)

- **B1 이중 소속 투수 8명** (extract.py:254): 같은 실제 투수가 injured/normal
  샘플로 동시 존재 → honest 변형에서 실제 투수 id 기준 분리 또는 제거.
- **B2 시대(era)·경력 교란** (extract.py:232,241): normal은 최초 4연속 시즌
  블록(과거 시대), injured는 수술 인접 시대에 앵커 + normal은 4연속 시즌
  생존자로 내구성 선택 편향 → era-matched 앵커/경력 분포 매칭 변형, 최소한
  시간적 홀드아웃으로 검증.
- **B3 right-censoring** (extract.py:227): 미래 TJS 투수가 normal에 포함
  가능 → 최신 레지스트리로 사후 N년 버퍼 라벨링(Phase 3의 +2024 확장과 결합).
- **B4 회귀 게이트 부재** (prepforreg.py:108): 회귀는 부상자만 학습 — 실전은
  분류기 게이트 뒤에서 작동. 2단계 파이프라인의 end-to-end 오차는 미측정
  → grouped split 공통 홀드아웃에서 연쇄 평가.
- **B5 KBO 창 실현성** (Prepfortrain.py:87): 224-bin(~3.3년) 밀집 이력은
  KBO에서 비현실적 → Phase 3에서 양 리그 동일한 짧은 창으로 재설계.

## C. 사소하지만 고칠 것 (LOW / 위생)

- **결과 CSV append 위험** (train_*.py): 변형 실험 전 **필수** — 변형별
  출력 경로 분리 + (model, seed) 중복 검사.
- 분류(100..1000)/회귀(102..1002) seed 불일치 → 변형에서 통일.
- L1이 pretrained backbone 전체에 적용 + l1_reg=0이어도 그래프 생성(낭비)
  → head-only + guard 변형.
- 4.7σ 아웃라이어가 diff_ax_CH 건너뜀 / 전체 데이터로 fit(경미 누수)
  → honest 변형에서 train-only fit + 첫 컬럼 포함.
- ViT 입력의 ~54%가 -5 상수 패딩(ImageNet 정규화 부재) — 클래스 편향은
  없음(전 샘플 동일), 성능 개선 여지로만 변형 실험.
- 시간 인덱스 컬럼이 feature로 유입(F=103, 논문 102) → 제거 변형.
- 스케줄러 첫 epoch lr≈0(죽은 epoch), 회귀 early-stop 반환 가중치(디스크
  체크포인트는 정상이라 잠복), step-fill vs 논문의 "linear interpolation"
  불일치 → 진짜 선형보간 변형.
- 기각 2건: zero_division=0(수치 동일), --shap-only RNG(영향 없음 확인).

---

## 실행 설계 (승인 대기)

**측정 전략**: 변형별 전 모델 재학습은 낭비 — **ViT(주력) + 회귀 1D-CNN**으로
변형 효과를 측정하고, 최종 확정 조합만 5모델 전체 재실행. ViT 10 seeds ≈ 30분,
회귀 10 seeds ≈ 15분(SHAP 제외) → 변형당 GPU ~45분.

| 단계 | 내용 | GPU |
|---|---|---|
| P2-0 | 위생: 변형별 결과 경로, seed 통일, 이중 소속 8명 처리 | 없음 |
| P2-1 | 회귀 GroupShuffleSplit(투수 단위) → honest R² | ~15분 |
| P2-2 | 분류 bug-fix 변형 (deepcopy + ViT shuffle) | ~30분 |
| P2-3 | fill-artifact 기여도 측정 (꼬리 마스킹/셔플 대조) | ~1시간 |
| P2-4 | 전처리 변형 (F=102, 진짜 선형보간, train-only 통계) | ~1시간 |
| P2-5 | 시간적 홀드아웃 (≤2021 학습 / 2022–23 테스트) | ~45분 |
| P2-6 | 지표 재산정 (PR-AUC는 저장된 preds로 GPU 없이, threshold 선택 변형) + paired 검정 | ~30분 |
| P2-7 | causal diff 재구축(A4) → 재추출 + 재학습 | ~2시간 |

합계 GPU ~6시간 내외. **전향적 태스크 재정의(A5)는 규모상 별도 트랙**으로
분리 권장 (Phase 2.5 또는 Phase 3와 병합) — Kang과의 비교가능성을 벗어나는
새 실험 설계이기 때문.

**결정 필요**: ① 위 스코프(P2-0~7) 승인 여부 ② A5 전향적 재정의의 분리 여부
③ 착수 시점.
