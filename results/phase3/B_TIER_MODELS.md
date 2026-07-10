# B 블록 — feature-tier × 모델 ablation (2026-07-10)

질문: ① Tier-2 트래킹 feature(spin·extension·release point·구종 구성)가
공개 계층 baseline(M-role)을 유의미하게 올리는가. ② tree 모델(HistGBM)이
LR보다 나은가. 답: **둘 다 아니오** (아래 3중 확인).

## 설계

- 코호트·평가: `cohort_v2.parquet`, fold_main(test 2022-23), H∈{90,150} B=0.
  동결 프로토콜(m0doubleprime/role_models와 동일): LR = StandardScaler +
  LogisticRegression(balanced), fit train+valid; 투수-clustered bootstrap
  1000회(seed 0)를 전 셀이 공유 → paired ΔROC/ΔPR; event recall@k.
  **게이트**: F1×LR이 M-role anchor(H90 ROC 0.643680/PR 0.035973, H150
  0.643775/0.046995)를 1e-4 내 재현하지 못하면 중단 — 실측 1e-7 내 통과.
- feature 세트:
  - **F1** = M-role 6개: pc_chronic, pc_acute_dev, days_since_last,
    vel_trend, month, start_share (공개 계층).
  - **F2** = F1 + Tier-2 12개. FB=FF+SI, BR=SL+CU+FC를 경기 단위로 NaN-safe
    투구수 가중 풀링. 만성 수준 4(spin_fb, ext_fb, spin_br, br_share) +
    급성 편차 6(spin/ext/relz/spin_br trend = 급성창 − 전 시즌 baseline
    [fallback 그 이전 경기들], |relx drift|, br_share_dev) + 변동성
    2(velo_sd, relx_sd — 급성창 경기 내 sd의 가중평균).
  - **F3** = 콘텐츠-only: vel_trend + Tier-2 (사용량·dsl·역할·month 제거).
- 창 정의 2종: **달력창**(급성 30일/만성 90일 — b_tier.py) 및
  **경기창**(급성 최근 5경기/만성 최근 15경기 — b_tier2.py, `_g` 접미).
  달력창은 4·5월 결정 시점이 오프시즌을 돌아봐 NaN 35–49% → 경기창으로
  커버리지 교정(NaN < 8%). LR은 대치(trend→0, level/sd→fit-set 중앙값),
  HGB는 raw NaN(네이티브 처리).
- HGB: stage-1 그리드 8설정(depth{2,3}×lr{.03,.1}×iter{100,300},
  leaf 30, l2 1.0) — train에서 fit, valid(2021) ROC로 선택, train+valid
  재적합, test 1회 평가.

## 결과 — 셀 (test ROC, H=90 / H=150)

| feature 세트 | LR | HGB |
|---|---|---|
| F1 공개 (M-role) | **0.6437 / 0.6438** | 0.6489 / 0.6431 |
| F2 달력 T2 | 0.6505 / 0.6272 | 0.5772 / 0.5615 |
| F2g 경기창 T2 | 0.6420 / 0.6327 | 0.5773 / 0.5663 |
| F3g 콘텐츠-only | 0.5453 / 0.5413 | 0.5363 / 0.4883 |

PR-AUC·event recall·CI는 `b_tier_cells.csv`, `b_tier2_cells.csv`.

## 결과 — paired 대조 (ΔROC [95% CI], M-role 공유 bootstrap)

| 대조 | H=90 | H=150 |
|---|---|---|
| T2 증분, 달력, LR | +0.007 [−0.043, +0.058] | −0.017 [−0.051, +0.020] |
| T2 증분, 경기창, LR | −0.002 [−0.052, +0.049] | −0.012 [−0.046, +0.026] |
| T2 증분, 관측 부분집합*, LR | +0.026 [−0.060, +0.110] | −0.009 [−0.067, +0.052] |
| T2 증분, HGB (달력/경기창) | −0.072 EXCL0 / −0.072 | −0.082 EXCL0 / −0.076 EXCL0 |
| 모델 효과 F1 (HGB−LR) | +0.005 [−0.046, +0.058] | −0.001 [−0.041, +0.039] |
| 콘텐츠-only vs M-role | −0.10 ~ −0.15 (대부분 EXCL0) | 동일 |

\* 관측 부분집합 = 달력 T2가 실제 관측된 test 창(4,507/8,011; 양성 65/95).
부분집합 전용 clustered resample 재구축. 전체 대조표: `b_tier_deltas.csv`,
`b_tier2_deltas.csv`.

## HGB 그리드 경계 확인 (b_hgb_boundary.py)

stage-1 그리드가 항상 최보수 모서리(depth 2, lr .03)를 선택 → 그 너머로
확장(48설정: depth 1, lr .01, iter 50, leaf 100, l2 10 포함):
- F1: H90 test 0.6549(LR +0.011, 단 PR은 악화), H150 0.6328(−0.011) —
  비일관·소폭, 기존 n.s. CI(±0.05) 안.
- F2g: −0.058 / −0.087 — 정규화를 더 줘도 회복 불가.
- H150에서 valid 0.70이 test 0.63으로 전이 실패 → 그리드 확장은 selection
  noise만 태움(valid 양성 47의 한계). **추가 hyperparameter 탐색 근거 없음.**

## 판정

1. **Tier-2 증분 널.** 달력창 널 → 커버리지 희석 우려 → 경기창(커버리지
   ~100%) + 관측 부분집합까지 3중 확인, 전부 CI가 0 포함 또는 음수.
   존재하더라도 **ROC +0.05를 넘는 증분은 배제**(LR 증분 CI 상한
   +0.02~+0.06). Kang SHAP의 "spin/extension 중요" 서사는 회고
   아티팩트 프레임에서만 성립했던 것으로 정리.
2. **tree 널.** 동일 6 feature에서 HGB ≈ LR; feature를 넓히면 유의하게
   악화(ΔROC −0.07~−0.08 EXCL0). 신호가 저차원·근사선형(만성 사용량 +
   역할 + 간격)이라 tree가 얻을 상호작용이 없고, test 사건 52–56 규모에서
   과적합 비용만 발생.
3. **콘텐츠-only ROC 0.49–0.55.** 신호의 대부분은 사용량·역할·등판 간격.
   "사용량 감소는 구단이 이미 안다"는 우려에 대해, MLB 공개+트래킹 수준의
   콘텐츠 신호로는 그 정보를 대체할 수 없음이 실측으로 확인됨.
4. 사전 규칙(양성 결과만 rolling-origin 검증)에 따라 rolling-origin 생략.

## G3(제안서) 함의

트래킹 데이터 논거는 "MLB에서 증분 입증" → **불가**로 확정. 정직한 정식화:
"MLB 규모에서 Tier-2 증분은 검출 한계 이하(+0.05 초과 배제). KBO에서의
가치는 리그 차이·내부 라벨(밀한 부상 기록)과 결합해야 검정 가능 — 데이터
접근이 곧 검정력"으로 전환. M-role(공개 계층 6 feature)이 최종 모델 후보로
유지됨.

## 산출물

- 스크립트: `results/phase3/scripts/b_tier.py`(달력 T2 6칸),
  `b_tier2.py`(경기창 + 부분집합), `b_hgb_boundary.py`(확장 그리드)
- 결과: `results/phase3/b_tier_cells.csv`, `b_tier_deltas.csv`,
  `b_tier2_cells.csv`, `b_tier2_deltas.csv`, `b_hgb_boundary.csv`
- 의존: `data/prospective/{cohort_v2,game_features_v2}.parquet`,
  scratchpad `slim_games.parquet`(재생성: `results/phase26/scripts/` 참조)
