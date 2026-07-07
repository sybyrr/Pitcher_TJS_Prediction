# Phase 2 / 2.5 워크로그 (실행 종료 — 아카이브)

**상태: 2026-07-07 완료. 모든 결과·해석의 canonical 소스는
`phase2_results.md`** (이 파일은 실행 방법의 기록만 남긴 아카이브).
계획: `plan_progress.md` (v2, G1–G3). 발견 목록: `phase2_findings.md`.

## 실행된 것 (요약)

- 분류 변형 10종 (v0 G1 + v1~v9, ViT 10 seeds씩) + 회귀 변형 4종
  (r0std/r1/r1o/r2, 10 seeds씩) + Phase 2.5 전향 학습 (LR + ViT 3 seeds).
- 결과 CSV: `results/phase2/`. 변형 정의는 `src/run_phase2.py` /
  `src/run_phase2_reg.py`의 VARIANTS 레지스트리가 canonical.
- 데이터 산출물: `data/final_df_causal.csv`(expanding-mean diff),
  `data/cohort_meta.csv`(샘플별 앵커 날짜), `data/prospective/windows.npz`
  (14,123 윈도우 × 146 bins × 102, 양성 217).

## 재현/확장 방법

- 변형 추가: run_phase2.py VARIANTS에 항목 추가 → `python src/run_phase2.py
  <이름>` (완료 seed 스킵, 재개 안전. detached + tail-monitor 패턴).
- 전향 설계 반복: prospective_build.py의 WINDOW_DAYS/HORIZON_DAYS/
  eligibility 수정 → build → prospective_train.py.
- 학습 루프 fidelity: src/train_loop.py는 upstream ViT.train_model의 충실
  포트 (deepcopy만 토글) — 다른 quirk(비가중 valid BCE 선택, epoch>10 게이트)
  보존됨.

## 실행 중 이슈 기록 (재발 대비)

1. pybaseball nullable dtype → parquet 왕복 유지 → boolean mask NA 오염
   (extract.load_raw에서 numpy dtype 정규화로 해결 — Phase 0에서 발견).
2. 우리 재추출 데이터에서 4.7σ 아웃라이어가 특정 투수의 컬럼 전체를 NaN화
   → fill 불가 → X에 NaN → 학습은 되나 확률 NaN으로 평가 크래시.
   phase2_data.build_arrays에 global-mean fallback 추가 (저자 데이터 no-op).
3. Savant 응답 ParserError는 재시도로 통과 (download_statcast.py에 내장).
4. cohort_meta는 finalize의 마지막 2명 제거 이전 시점 기록이라 657행
   (final 656의 상위집합) — 앵커 조회 용도로는 무해.
