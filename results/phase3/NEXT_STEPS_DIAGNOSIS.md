# 진단 + 다음 스텝 설계안 (2026-07-10, ultracode 6-agent 조사 → fable 종합; 사용자 확정 대기)

성능(0.644)의 원인 진단과, 검증된 개선 레버의 실행 설계. 코드 미수정.
조사 원본: scratchpad `wf_results/*.json` (진단 3 + 문헌·GitHub·방법론 3, opus).
핵심 수치 2건은 fable이 재계산으로 검증 (`verify_diag.py`).

## 1. 진단 — 0.644는 어떻게 결정되는가

버그 없음: 게이트(anchor 1e-7)·fold·라벨·bootstrap·사건 그룹핑 전부 검증 통과.
성능은 3겹 구조:

**하드 리밋 (모델 교체로 못 넘음)**
- 사건 수 52-56 → paired ΔROC CI 반폭 ±0.05-0.06 = 검출 플로어.
- 불펜 구조적 불가: recall@50 포착 전원이 선발; 미스의 ~70%가 불펜.
  불펜 내 전 feature 단변량 AUC 0.44-0.52(우연 수준), 미스 불펜은
  mid-pack(중앙 percentile ~62-65) — 경기 데이터로 재순위 불가.
- catchability 천장 ~69% (≥20경기 적격성 + 정보적 검열: 수술자 65%가
  수술 전 30일+ 잠적, 중앙값 54일).
- 문헌 천장: 정직 전향 설정의 공개 최고치 0.61-0.67. 0.644는 그 안.

**교정 가능한 depressor (fable 재검증 완료)**
- D1 조기 시즌 구조적 0: 달력 90일 창이 오프시즌을 조회 → pc_90==0이
  전체 창의 36.1%, 4월 82%, 5월 40%; 2022-04-01은 test 662창 전부 퇴화;
  test 양성의 28-29%. 이때 pc_chronic·pc_acute_dev가 동시에 0, vel_trend도
  대부분 0 → 4-5월 순위가 dsl+start_share로 붕괴. 교정 프로브(전 시즌
  rate 백필, as-of 안전): 점추정 +0.010~+0.015 ROC, recall@50 13→16(H90).
- D2 vel_trend 결측→0 (전체 43%, 양성 45-46%): "측정 불가"와 "변화 없음"이
  동일값 — 결측 지표 부재. 셧다운 위험군을 안정 베테랑과 동일 취급.
- D3 2023 약세: test ROC 2022 0.688/0.673 vs 2023 0.602/0.614 — 0.644는
  강한 해와 약한 해의 혼합. 직접 교정 불가, test 연도 추가로 희석 가능.

**feature 기여 실상** (drop-one): dsl(−0.031~−0.041) > start_share >
month > pc_acute_dev; **pc_chronic은 현 인코딩으론 role 프록시**(제거 시
+0.003~+0.008, corr 0.46 w/ start_share), vel_trend 사망(±0.001). 단
D1·D2가 원인의 일부 → 삭제가 아니라 수리 대상. 선발 내 잔여 구조
존재(role-stratified 프로브 +0.029/+0.020, P(개선)=0.88/0.80, CI는 0 포함,
recall 방향 불안정).

**메트릭 주의**: role 도입이 ROC/PR을 올리며 H90 recall@50을 19→13으로
낮춤(prevalence 층화가 top-50을 선발로 재배열) — headline 지표 확정 필요.

## 2. 다음 스텝 설계안 (우선순위순; 전부 동결 프로토콜 + paired 대조)

**P0-1. 2024 시즌 확장** [borrow, 최우선] — test 2022-23 → 2022-24
(train/valid 동결, test 순증). 예상 +26/+28 test 사건(52→~78, +50%),
CI 15-20% 축소, 제3 test 연도로 D3 희석. 검증 완료: plate_x/z 정의 변경은
2025+(2024 무관, M-role은 plate 미사용), 피치클록은 2023부터(연속성),
라벨은 라이브 시트(~2026)로 성숙. **비용: statcast_2024.parquet 미보유 —
다운로드 선행 필요.** 이후의 모든 검정이 넓어진 사건 기반 위에서 판정됨.

**P0-2. M-role v2 구조 교정** [진단 직결] — (a) season-aware chronic:
조기 시즌 pc_chronic을 전 시즌 rate로 백필/블렌드 + season-fresh 지표
(b) vel_trend 결측 지표. 6→최대 8 feature. 사전 등록: paired ΔROC/ΔPR +
recall@50 + rolling-origin 방향 일관성. 기대 +0.01~0.02 (프로브 실측 기반).

**P1. 신규 feature 미니 티어 3종** (각각 1-3개, 개별 additive paired 검정;
8 feature 초과 조합은 Firth/logF 병용 — Puhr 2017/Rahman 2017 근거):
- (a) **과거 TJS 이력 flag** [미검정 신규] — fable 검증: 코호트 투수의
  36%가 t 이전 TJS 보유, test base rate **보호 방향 0.55×**(재수술 통념과
  반대 — survivor/새 인대 효과 해석). 1.8× 구배는 role(2×)급. as-of 안전
  (t 이전 수술만). 주의: 시트의 과거(아마추어·마이너) 수술 소급 완전성.
- (b) **pitch-mix 단독 재검정 + torque-가중 부하** — breaking share는
  Tier-2 12개 블록에 묻혀 검정됨 → 2-3개 단독 재검정(Karnuta SHAP
  slider-usage 근거). torque-가중 pc_chronic(구종별 팔꿈치 토크 가중,
  Driveline 문헌 상수로 사전 등록, tuning 금지).
- (c) **rest 구조** — 단기휴식(<4일) 등판 수·간격 분산 (trailing 30-90d).
  pc_acute_dev와 중복성 사전 점검.

**P2. 모델 형태** (지표 기대 낮음, 정당성·일관성 목적):
- discrete-time hazard 재정식화 [adapt] — 현 42 결정일 설계가 곧 person-
  period/landmark 구조(Suresh 2022, van Houwelingen). 수술당 ~2.3개 상관
  양성창의 이중계상 제거, H90≤H150 정합 보장, 제안서 방법론 격상. 무료
  차용: "landmark supermodel" 명명.
- role-stratified 보정 1회 사전 등록 검정 (+0.02-0.03 프로브, 불안정).
- (선택) Bayesian 보고층 — 계수 posterior로 소표본 불확실성 전달.

**기각 (근거)**: 시퀀스 DL(Christodoulou 2019 — LR 대비 무이득, DL 우위는
~천 단위 양성부터; 우리는 distinct 사건 ~77), nested case-control(full
cohort 이미 사용 — 검정력 상승 불가), HGB 추가 탐색(경계 확인 종료),
기계적 Tier-2 재방문(3중 널; upstream handedness 정규화 우려는 우리 설계에
비적용 — trend는 투수 내, level은 부호 무관), full GAM(EPV), 경쟁 레포의
IL 라벨·랜덤 split(누수 반면교사: AMesa2 AUC~1.0 = 순환 라벨).

**기대치**: 현실 목표 = ROC 0.65-0.68 + CI 축소 + 3년 test. 문헌 천장
0.61-0.67. 0.75+로 가는 경로는 MLB 공개 데이터에는 없음(불펜 하드 리밋).

## 3. 사용자 결정 3건
1. P0→P1→P2 순서 승인 여부 (일부 발췌 실행도 가능).
2. headline 지표 확정 — 권장: H150 ROC + event recall@50 병기 (H90은
   임박 확인 보조).
3. 2024 Statcast 다운로드 착수 승인 (P0-1 전제, 수 시간).
