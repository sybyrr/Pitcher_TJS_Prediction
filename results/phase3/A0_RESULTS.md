# A0 블록 — 평가 교정 잔여 3종 (2026-07-13, 사양 v2)

> **구현 보수 완료 (2026-07-14, M1-A0; 아래 2026-07-13 원문보다 우선):**
> 과거 `a0_recal.csv`와 `a0_sens.csv`는 감사 이력으로 그대로 보존했다.
> 보수 코드는 각각 `a0_recal_corrected.csv`,
> `a0_recal_oof_window.csv`, `a0_sens_corrected.csv`를 새로 생성한다.
> canonical 모델·계수·fold·평가 경계·대표 ROC는 변경하지 않았다.
>
> ① 과거 `citl` 열은 표준 calibration intercept가 아니라
> `mean_pred - prevalence`였으므로, corrected 산출물에서는 **`mean_error`**로
> 명명했다. 표준 보정 절편은 최종 window `P(H)`에 대해
> `logit(Y)=a+1×logit(P)`의 비벌점 MLE인
> `cal_intercept_fixed_slope1`로 별도 계산했다. 절편·기울기 동시 비벌점
> 적합도 `cal_intercept_joint`, `cal_slope_joint`로 분리했다.
> ② 투수-grouped 5-fold OOF 최종 `P(H)` 진단을 최초 영속화했다.
> H90/H150 모두 n=16,949(투수 1,029명), 양성 window 173/252개다. 평균 예측−
> 관측은 −0.000170/−0.000161, 표준 보정 절편(slope=1)은
> **+0.0170/+0.0111**로 전체 평균은 가깝다. 그러나 joint slope는
> **0.2658/0.5683**으로 1보다 작다. 따라서 interval 재보정의 항등성이나
> 평균 일치만으로 최종 window 확률의 보정을 주장할 수 없다. OOF
> `P90>P150` 위반은 0건이다.
> ③ mature test raw `P(H)`의 표준 보정 절편은 H90/H150
> **+0.363/+0.308**, joint slope는 **1.940/1.962**다. 기존 본문의
> 1.86/1.90은 기본 L2가 적용된 sklearn slope였으므로 표준 비벌점 진단값은
> corrected CSV가 정본이다. 이는 절대확률 과소예측을 재확인하지만 원인을
> 유병률 shift 하나로 단정하지 않는다. ROC 0.7009/0.6958은 불변이다.
> ④ outcome-dependent `one_landmark_per_surgery`는 수술일로 가장 가까운
> 양성행을 고른 역사적 stress로만 유지한다. 새 outcome-blind 규칙은
> **라벨·수술일을 보기 전에 (투수, 달력연도)별 첫 적격 결정일 하나를
> 선택하고 그 window의 모든 person-period 행을 보존**하는 것이다.
> 이 규칙으로 fit만 축소하면 3,251 windows/양성 interval 55개,
> all-landmark test ROC는 **0.6225/0.6250**(canonical 대비
> −0.0783/−0.0708)이다. 같은 규칙의 test(2,308/2,290 windows,
> 양성 window 42/64개)에서 canonical fit은 0.6495/0.6435, 단일-landmark fit은
> 0.5642/0.5850이다(동일 평가집합 대비 −0.0853/−0.0585). 이는 선택 규칙과
> estimand가 다른 민감도이며 canonical 점수를 대체하지 않는다.
> Corrected SHA-256: recal
> `583b9a4c7c746c9e505d00848a46b99bce09d0f2edf467e4513e88dd8f1cf10d`,
> OOF window
> `5474cda20919d56965d900e228d906b6d1abbf5236bf2c322a030538d968c83c`,
> sensitivity
> `5b20dbef3b4c9d9ea56e402282d9c119aa4c89b3c6f633ab30951e8950e69274`.

> **정정 (2026-07-13, codex 3차 감사 수용):**
> ① A0-1: fold 3개에 낙관이 −0.030/+0.073/−0.081로 흔들리므로 평균
> −0.013으로 **"체계적 낙관 없음"을 주장할 수 없고, 전체 adaptive
> 탐색 낙관의 수치적 "하한"도 아니다** (범위가 3후보로 제한된 감사일
> 뿐). 정확한 표현: "사전 고정 3후보 감사에서 평균 낙관은 작았으나
> fold 변동이 커서 낙관 유무는 확정 불가; 전체 탐색 낙관은 미측정."
> ② A0-2의 `citl`은 표준 calibration-in-the-large(logit 절편)가 아니라
> **mean-pred − prevalence** — 명칭만 정정, 수치는 유효. ③ 구간 수준
> 재보정 계수가 항등(b≈0.97)이어도 **최종 출력 P(H)의 window 수준
> 보정을 의미하지 않는다** (fit-set window OOF slope 0.24/0.55). ④ test
> slope ~1.9의 원인을 유병률 shift로 **단정할 수 없다** — shift와
> 정합하지만 조건부 분포 변화·모델 오지정·landmark 구조를 배제하지
> 못함. 운영 결론(절대 확률 인용 금지, 순위만 사용)은 불변. ⑤ A0-3의
> one_landmark는 제거된 landmark의 선행 음성행을 남기므로 순수 물리적
> dedup이 아니라 **outcome-dependent stress test**로 읽을 것. 본문은
> 이력 보존을 위해 유지.

사양: `plan_progress.md` "2026-07-13 (계속 3)" (codex 2차 감사 반영).
전부 **보고 전용** — canonical 수치(H90 0.701 / H150 0.696, 조건부
backtest, safety 경계 0.660/0.665 병기)는 불변. 코드
`scripts/{a0_nested,a0_recal,a0_sens}.py`, 수치 `a0_nested.csv,
a0_recal.csv, a0_sens.csv`.

## A0-1 — nested 3-후보 안정성 감사 [철회된 2026-07-13 원문]

fold Y별 inner(fit ≤Y−2, valid Y−1)에서 등록된 선택 경로(이진
{M-role, M_bf, M_sa} 최고 → hazard 비열등 −0.005)를 자동 재실행,
years ≤Y−1 재적합 후 outer 연도 Y 평가. 낙관 = inner 추정 − outer 실측.

| Y | 선택 | 형태 | H90 inner→outer (낙관) | H150 inner→outer (낙관) | 고정 M_sa+hazard outer (H90/H150) |
|---|---|---|---|---|---|
| 2022 | M-role | hazard | 0.634→0.664 (−0.030) | 0.640→0.668 (−0.028) | 0.672 / 0.680 |
| 2023 | M_sa | binary | 0.692→0.619 (+0.073) | 0.688→0.622 (+0.065) | 0.641 / 0.639 |
| 2024 | M_bf | hazard | 0.645→0.726 (−0.081) | 0.643→0.718 (−0.075) | 0.777 / 0.778 |

- **3-fold 평균 낙관 −0.013 (양 H 동일)** — 사전 고정 3후보 내 선택에는
  체계적 낙관이 없고, 연도 간 변동(±0.07~0.08, fold당 사건 23-29)이 지배.
- 선택은 fold마다 다름(M-role→M_sa→M_bf) = 선택 불안정성 실증. 단,
  **고정 M_sa+hazard가 모든 fold에서 fold별 선택보다 우수**(차 −0.009~
  −0.060) — canonical 선택이 이 후보군 안에서는 사후에도 정당화됨.
- 한계(사전 명시): 이는 3후보 감사이며 실제 P-블록 탐색(수십 변형)의
  낙관을 포괄하지 않는다 — **하한으로만 인용**.

## A0-2 — 투수-grouped OOF, hazard 구간 수준 공동 재보정 [철회 표현 포함 원문]

- fit set(2017-21) 투수-grouped 5-fold OOF → 구간 로지스틱 재보정 계수
  **a=−0.164, b=0.967 ≈ 항등** (temporal LOYO: a=−0.162, b=0.955).
  → **fit 시대 내부에서 hazard 확률은 이미 보정돼 있음.**
- 그런데 mature test slope는 raw 1.86/1.90 → 재보정 후에도 1.92/1.96
  (mean-pred 1.00%/1.50% vs 유병률 1.40%/1.97%). **test 미보정의 원인은
  과적합이 아니라 2017-21→2022-24 유병률 상승(시대 shift)** — fit-set
  기반 재보정으로는 교정 불가능.
- **운영 규칙 유지·강화: 절대 확률은 배치 시점의 최신 라벨로 재보정하기
  전에는 인용 금지. 순위 지표는 영향 없음** (ROC 0.7009/0.6958 불변).
- P90≤P150 정합: joint hazard 재보정 위반 0. marginal Platt(참고)도
  위반 0이나 계수가 심하게 shrink(b 0.24/0.55, window 클러스터 탓) —
  joint 방식만 사용.
- safety 경계(2024-06-30)에서도 동일 패턴 (slope 1.50/1.68).

## A0-3 — hazard cluster/형태 민감도 [철회 표현 포함 원문]

| 변형 | H90 | H150 | Δ |
|---|---|---|---|
| canonical (plain, 선형 s) | 0.7009 | 0.6958 | — |
| s 범주형 | 0.7008 | 0.6958 | ±0.000 — 선형 s 충분 |
| 사건 가중 (양성 구간 1/landmark) | 0.6725 | 0.6696 | −0.028/−0.026 |
| 수술당 단일 landmark (양성 행 173/252 제거) | 0.5965 | 0.5923 | **−0.104** |
| cluster full-refit bootstrap (B=200, fit 투수 재표집) | mean 0.668, CI [0.597, 0.702] | mean 0.664, CI [0.600, 0.700] | — |

- 반복 landmark 양성 구조가 canonical 성능의 실질 일부다: 가중 제거 시
  −0.03, 물리적 dedup 시 −0.10 (남는 양성 79행, EPV 붕괴 포함).
- **fit-side 불확실성만으로 CI 하한 ~0.60** — test-side bootstrap CI
  (0.643-0.759 / 0.645-0.746)와 별개 축. 정직 표현: "조건부 backtest,
  경계·적합 불확실성 포함 실질 대역 ~0.60-0.70"이 A0 전체로 재확인됨.

## 종합

- canonical 수치·모델 변경 없음. 인용 시 (i) 조건부 backtest 표기,
  (ii) safety 경계 0.660/0.665 병기, (iii) 절대 확률 재보정 전 인용
  금지, (iv) 낙관 감사는 "3후보 하한 −0.013, 연도 노이즈 지배"로 인용.
