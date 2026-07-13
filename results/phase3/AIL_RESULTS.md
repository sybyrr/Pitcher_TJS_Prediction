# A-IL 블록 — 팔꿈치-IL 이력 feature (2026-07-13, 사양 v2)

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

## 판정 요약

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
