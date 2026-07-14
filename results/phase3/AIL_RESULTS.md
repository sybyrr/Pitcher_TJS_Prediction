# A-IL 블록 — 팔꿈치-IL 이력 feature (2026-07-13, 사양 v2)

> **구현 정정 완료 (2026-07-14, M1 A-IL 감사 보수):** 아래 2026-07-13
> 본문과 `ail_results.csv`는 이력 보존을 위한 **legacy 결과**다. 이후
> transaction의 elbow/수술 상태를 episode 시작으로 소급하던 parser,
> transfer/activation gap 처리, surgery-date 기준 history/lead를 코드에서
> 보수했다. 정정 정본은 `il_episodes_asof_v2.parquet`,
> `ail_results_asof_v2.csv`, `ail_alert_events_asof_v2.csv`다. **TJS-only
> label과 canonical 미채택은 변경하지 않는다.**

## 2026-07-14 corrected as-of 결과

### 결정

**canonical 미채택 유지.** 전체 공개 IL 신호의 증분은 재현되지만 30–90일
blackout 증분은 모두 0과 양립한다. 60일 점추정은 정정 뒤 H90/H150 모두
미세한 양수(`+0.00577/+0.00232`)가 되어 legacy의 사전 방향-only 조건을
기계적으로 만족한다. 그러나 이는 결과를 이미 본 뒤 parser 정의를 바로잡은
감사 결과이므로 소급 승격 근거로 사용하지 않는다. 95% CI는 두 horizon 모두
0을 포함하고 90일 H90은 다시 음수다. 분류는 계속 **공개 진단 후 triage
재검정 후보**이며, 조기경보 canonical feature가 아니다.

### 구현 정정

- episode에는 `elbow_disclosure_date`, `post_tjs_disclosure_date`, 더 넓은
  `post_procedure_disclosure_date`를 별도로 보존한다. t 시점 feature는
  `disclosure_date < t`인 정보만 사용한다. 뒤늦은 transfer의 elbow/TJS
  문구는 episode `start`로 소급되지 않는다.
- placement·transfer·activation **모두** 현재 open episode의 마지막
  action과 60일 초과 gap인지 먼저 검사한다. 초과 시 `gap_timeout`으로
  닫고, transfer는 새 공개 episode를 열며 activation은 orphan으로
  기록한다. activation이 60일 이내면 `activation`으로 닫는다.
- ontology는 세 층으로 분리했다: 명시적 Tommy John, UCL surgery/
  reconstruction/repair, 기타 완료 elbow procedure. feature의
  `new_elbow` 상태는 당시 공개된 완료 procedure를 제외한다. 미래 의도
  (`scheduled`, `recommended`, `will`, `may need`)와 thumb/finger/wrist UCL은
  제외한다.
- 내장 회귀 예: `Recovering May ... Tommy John surgery`, `Tommy John
  surgery recovery`, `Right UCL surgery rehab`은 완료 procedure true;
  `UCL sprain`, `Tommy John surgery recommended`, `Scheduled to undergo ...`,
  `Right thumb UCL repair`는 false다. ontology 25개와 synthetic episode
  disclosure/gap/closure 검사가 통과했다.
- blackout은 elbow/any-IL 공개일을 t−X 이전으로 제한하되, t 시점에 이미
  공개된 post-procedure 상태를 숨겨 과거 elbow episode를 되살리지 않는다.

### corrected 데이터 gate

- 입력 action은 legacy와 동일한 **13,827행**(place 7,221 / transfer 1,145 /
  activate 5,461)이다.
- 일관된 gap 처리 후 **6,816 episode**: activation closure 4,101,
  gap-timeout 2,031, right-censored 684. 누락/장기 gap 뒤 transfer가 연
  episode 143, gap 뒤 unmatched activation 1,360은 별도 audit count다.
- elbow 1,079, 명시적 TJS/UCL procedure 140, 모든 완료 elbow procedure
  183, final-state new-elbow 896이다. final-state 수치는 설명용이며 실제
  feature는 각 decision date의 as-of 상태를 다시 계산한다.
- elbow가 episode 시작 뒤 처음 공개된 경우는 **36건**(lag 중앙값 20일,
  최대 88일). legacy 감사의 33건/최대 773일과 차이가 나는 이유는 60일
  초과 transfer를 이전 episode에 붙이지 않고 새 episode로 분리했기 때문이다.
- cohort window coverage는 elbow2y 16.03%, anyil2y 61.40%다.

### corrected paired 결과

동결 fold, mature 경계 `t+H≤2024-12-31`, 투수-clustered bootstrap
1,000회(seed 0), legacy A-IL diagnostic LR 형태를 그대로 사용했다.

| 항목 | H90 | H150 |
|---|---:|---:|
| 전체 ROC (M_sa→M_il) | 0.6920→0.7605 | 0.6917→0.7361 |
| 전체 dROC | **+0.06862 [+0.01280,+0.12308]** | **+0.04453 [+0.00151,+0.08845]** |
| 전체 dPR | **+0.07883 [+0.04087,+0.12750]** | **+0.07120 [+0.03931,+0.11546]** |
| blackout 30d dROC | +0.00775 [−0.04669,+0.06572] | −0.00075 [−0.03731,+0.04019] |
| **blackout 60d dROC** | **+0.00577 [−0.02787,+0.04803]** | **+0.00232 [−0.02067,+0.02990]** |
| blackout 90d dROC | −0.00614 [−0.03920,+0.03005] | −0.00011 [−0.01950,+0.02101] |

### caught-alert 기준 history와 lead

각 사건에서 M_il top-50에 처음 포착된 decision date를 `first_il_alert`로
고정한 뒤, 그 alert **직전 공개돼 있던** elbow history만 조회했다. 수술일
이전 아무 episode나 사후 검색하는 legacy 계산은 사용하지 않았다.

| 항목 | H90 | H150 |
|---|---:|---:|
| event recall@50 (M_sa→M_il) | 24→41 / 75 | 23→43 / 80 |
| 신규 포착 / 상실 | 26 / 9 | 25 / 5 |
| caught alert 당시 elbow history 없음 | 0/41 | 2/43 |
| history 공개→수술 lead, 중앙값 [P25,P75] | 56 [25,74]일 (n=41) | 56 [25,83]일 (n=41) |
| 첫 caught alert→수술 lead, 중앙값 [P25,P75] | 36 [16,53]일 | 37 [16,68]일 |

따라서 정정된 공개-history lead는 약 56일이지만, 모델의 실제 첫 caught
alert는 중앙값 36–37일 전이다. 전체 신호의 대부분이 임박한 공개 elbow
진단을 재포장한다는 해석은 유지된다. legacy의 "잡힌 사건 중 history
전무 0%"는 H90에는 맞지만 H150에는 2/43 예외가 있어 철회한다.

### 재현 산출물

- parser: `scripts/ail_parse.py` →
  `data/ail/il_episodes_asof_v2.parquet` (SHA-256
  `f1aea9d48297acbc8814cb25cca50e501fc6d06818d90e5f1ef7eb3c443eeb15`)
- 평가: `scripts/ail_eval.py` → `ail_results_asof_v2.csv` (SHA-256
  `9943dacb6aa217dbf9071f45c1e04bb16c769e01f14f109d6a9bfb352cff9734`) +
  `ail_alert_events_asof_v2.csv` (SHA-256
  `8b49fcc898097163d6ec12be280661c596a78f50159fa778a8a690b00710c327`)
- `.venv` Python으로 parser/eval 내장 테스트와 `py_compile` 통과. parser
  parquet 및 두 CSV는 연속 재실행에서 byte-identical hash를 확인했다.
  legacy `il_episodes.parquet`와 `ail_results.csv`는 덮어쓰지 않았다.

> **정정 (2026-07-13, codex 3차 감사 수용 — fable 재계산 검증):**
> parser 한계 3종 확인. ① **elbow 플래그 소급**: episode 내 이후
> transaction(transfer 등)의 텍스트가 episode 시작일로 소급됨 —
> 1,033개 중 **33개** (최대 773일, 중앙값 33일; fable 재계산 일치).
> ② transfer/activation에는 60일 gap 규칙이 적용되지 않아 episode가
> 무기한 연장될 수 있음. ③ post-TJS 회복 regex("recovering from ...
> tommy john")가 다른 표현을 놓침. **첫 공개일(disclosure date) 기준으로
> 정정한 blackout 60d dROC = fable −0.002/+0.001 (codex −0.0005/
> +0.0009; 구현 차이 내 동일)** — 어느 쪽이든 ≈0으로 승격 게이트 실패는
> 불변, **canonical 미채택 결론 유지**. lead도 경보 시점 이용 가능 정보
> 기준으로는 중앙값 ~54-56일(codex; 본문의 34일은 소급 포함 수치) —
> 여전히 B'의 경기 데이터 lead 72일보다 짧은 진단-echo 대역. 아래
> "합법적 용도는 triage" 표현은 **"추후 재검정이 필요한 triage 후보"**로
> 정정 (parser 수정 전 검증된 모델 아님). 본문은 이력 보존을 위해 유지.

사양: `plan_progress.md` "2026-07-13 (계속 3)". 라벨은 TJS-only 불변 —
IL은 입력 feature만. 코드 `scripts/{ail_fetch,ail_parse,ail_eval}.py`,
수치 `ail_results.csv`, 데이터 `data/ail/` (gitignore).

## 판정 요약 [legacy parser의 철회된 2026-07-13 원문]

**canonical 미채택.** 전체-feature 개선은 크지만(+0.068/+0.046 EXCL0),
사전 등록 blackout 게이트(60일에서 방향 유지)를 **실패** — 신호의 실체는
"이미 공개된 팔꿈치 진단의 메아리"였다. 분류: **공개 진단 후 triage
신호** (조기경보 아님).

## 데이터·게이트 (성능 조회 전 동결)

- MLB StatsAPI transactions 2016-2024 (월별 fetch, `data/ail/*.json`).
  정보시점 = transaction `date` (소급 effectiveDate 미사용). MLB 30팀
  toTeam 필터로 마이너 혼입 제거.
- IL 행 13,827 (place 7,221 / transfer 1,145 / activate 5,461) →
  episode 병합(placement 개시, 60일 내 중복/transfer 흡수, activation
  종결) → **episode 6,692, 팔꿈치 1,067, post-TJS 회복 34** (신규 팔꿈치
  = 1,033). 연도별 521-1,027 (2020 저점) — sanity 통과.
- 동결 키워드: `elbow|forearm|ucl|ulnar|tommy john` (대소문자 무시);
  "recovering from ... tommy john"은 신규 팔꿈치에서 제외(별도 플래그).
- 무작위 40건 수동 검수: **분류 정밀도 40/40** (ELBOW 7건 전부 실제
  팔꿈치/전완). 알려진 재현율 한계: "flexor strain"이 동결 셋에 없어
  미포착 — 셋은 사전 동결이므로 유지, 한계로 기록.
- coverage: 창의 17.6%가 최근 2년 팔꿈치-IL ≥1, 61.1%가 임의-IL ≥1.

## 결과 (M_il = M_sa + {elbow2y, dsle_log, anyil2y}, mature test)

| 항목 | H90 | H150 |
|---|---|---|
| 전체 feature dROC | **+0.068 [+0.009, +0.125] EXCL0** | **+0.046 [+0.001, +0.092] EXCL0** |
| 전체 feature dPR | +0.073 EXCL0 | +0.067 EXCL0 |
| recall@50 | 24→44 (신규 29, 상실 9) | 23→45 (신규 27, 상실 5) |
| blackout 30d dROC | +0.006 [−0.043, +0.063] | +0.002 [−0.034, +0.047] |
| **blackout 60d dROC (승격 기준)** | **−0.006** | **−0.003** |
| blackout 90d dROC | −0.015 | −0.005 |

- **lead 분해가 원인 확정**: 이력 보유 사건의 마지막 팔꿈치-IL→수술
  간격 중앙값 **34일** (P25 13일, P75 ~72일). M_il이 잡은 사건 중
  팔꿈치-IL 이력이 전무한 사건 = **0%**. 즉 개선분은 거의 전부
  "수술 직전 IL 등재"라는 공개 정보의 재포장이다.
- 주 계수도 dsle_log −0.63/−0.55 (최근일수록 위험)에 집중 — 같은 결론.

## 함의

- **조기경보 canonical에는 넣지 않는다** (60d blackout에서 널-음수).
  경기 데이터 모델(M_sa+hazard)의 가치는 IL 공개 "이전" 신호라는 점이
  유지된다 (B' 블록: 경보의 60%가 사용량 정상 시점, lead 중앙값 72일).
- 합법적 용도는 **triage**: 이미 팔꿈치-IL에 오른 투수의 수술 전환
  위험 순위화. 배치 시 별도 계층(공개 진단 후)으로만 문서화.
- KBO 제안서: IL 이력을 조기경보 성능의 근거로 인용하지 말 것. prior_tjs
  계열(P1 열린 후보)과 동일한 "구단이 이미 아는 정보" 주의가 적용된다.
