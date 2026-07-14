# KBO 적용 가능성 검토

> **상태 정정 (2026-07-13):** 이 문서의 초기 Tier-2 협상 서사는 후속 B/B'
> 실험 전 가설이다. 달력창·경기창·관측 부분집합 모두에서 Tier-2의 일관된
> 증분을 검출하지 못했으므로, "release/extension/spin axis가 MLB 성능을
> 높였다"고 주장하지 않는다. 현재 KBO 요청 논거는 raw tracking 자체의
> 입증된 증분이 아니라, 내부 TJS/UCLR 라벨·실제 역할·숨은 투구량과 결합해
> 증분을 검증할 수 있는 데이터 계약이다. MLB 계수와 primary 경계 novel
> ROC 0.66을 KBO 성능으로 이식하지 않는다. **현재 canonical 결론은 문서
> 끝의 "2026-07-13 v2 정정"이며, 아래 초기 본문의 tracking-first·공개
> 적용 불가·Tier-2 협상 문구는 역사적 가설로만 읽는다.**

작성 기준일: 2026-07-01. 근거: KBO 트래킹 인프라·데이터 공개 현황 및 부상자 명단 제도 검색.
전제: 앞 단계에서 MLB 모델(Kang 재현 + short-term distance 지표)을 먼저 만든 뒤, KBO로 이전하는 시나리오.

---

## 한 줄 결론

**기술적으로 KBO 구장에는 트랙맨이 깔려 있어 원리상 같은 종류의 데이터가 생성되지만, 그 데이터가 외부에 공개되지 않아 구단 협조 없이는 접근 불가하다.** 즉 KBO 적용의 병목은 "feature가 부족해서"가 아니라 "데이터 접근 자체가 막혀 있어서"다. 이건 사용자의 구단 컨택 전략과 정확히 맞물린다 — 컨택의 목적이 "없는 feature를 만들어달라"가 아니라 "있는 데이터를 열어달라"가 되어야 한다.

---

## 1. 트래킹 인프라 — 있다 (MLB와 같은 계열)

- **트랙맨이 KBO 1군 대부분 구장에 설치**되어 있고, 일부 2군 구장·아마추어 목동구장에도 있다. 트랙맨은 MLB Statcast의 이전 세대(2015~2019) 기반 기술과 같은 레이더 방식이다.
- 광주 기아 챔피언스필드에는 **호크아이**(현재 MLB Statcast 기반 장비)도 유일하게 설치·운용 중.
- 2025년부터 KBO 중계에 트랙맨 기반 트래킹 데이터(구종, 구속, 회전수 등) 시각화가 시범 도입됐다(잠실·광주 한정 시작).
- **ABS(자동 볼판정)는 트랙맨이 아니라 스포츠투아이의 PTS 기반**이고, PTS는 PITCH f/x 계열 카메라 방식이라 트랙맨/호크아이와 **측정값 편차가 있어 서로 직접 비교가 어렵다.**

→ 함의: 원리상 Kang의 102 feature 중 상당수(구속, 회전수, release position, extension, spin axis 등)를 **생성할 수 있는 장비는 존재**한다. 문제는 그 다음이다.

## 2. 데이터 공개 — 막혀 있다 (여기가 진짜 병목)

- MLB는 Baseball Savant를 통해 pitch-level 데이터를 **무료 공개**하고 pybaseball로 누구나 받는다. **KBO에는 이에 상응하는 공개 창구가 없다.**
- KBO 공식 기록실은 전통 스탯(구속 일부 포함) 수준의 집계 기록만 제공하고, **pitch-by-pitch 트래킹 원자료는 공개하지 않는다.** 트랙맨/호크아이 원자료는 각 구단·리그·중계사가 보유하며 외부 비공개다.
- 국내 세이버 커뮤니티에서도 지적되듯, 고급 트래킹 데이터는 구단·업체에 귀속되어 대중 공개가 사실상 끊긴 상태다.

→ 함의: **구단(또는 스포츠투아이/트랙맨코리아) 협조 없이는 pitch-level 데이터 확보가 불가능하다.** 이것이 KBO 적용의 1차 관문이며, feature 설계보다 앞선 제약이다.

## 3. 부상 라벨 — 부분적으로만 존재

- KBO도 **부상자 명단(IL) 제도**를 운영한다(투수 15일 등). 다만 MLB의 60일 IL / 공개 injury database처럼 **정형화·공개된 TJS 전용 데이터셋은 없다.**
- 개별 TJS 사례는 언론·구단 발표·백과 문서로 파편적으로 확인된다(예: 곽도규, 강재민, 이의리, 문동주 관련 보도 등). 즉 **라벨을 만들려면 언론/구단 발표를 수작업 집계**해야 한다 — Roegele 시트가 없는 상황.
- 리그 규모가 MLB의 1/5 수준(10구단)이라 **연간 TJS positive 수가 절대적으로 적다.** 이건 앞서 논의한 "distance-based가 few-shot에 유리한 이유"와 직결되지만, 동시에 "validation용 positive가 부족하다"는 한계도 그대로다.

## 4. feature 이전 가능성 — 계층별로 다름

MLB 모델을 KBO로 옮길 때, feature가 살아남는 정도가 계층적으로 갈린다. (트랙맨 데이터 접근이 된다는 가정 하에)

| feature 계층 | KBO 이전 가능성 | 근거 |
|---|---|---|
| 구속(release_speed), 회전수(spin_rate), 구종 | 높음 | 중계에도 표출될 만큼 트랙맨이 안정 측정 |
| release position, extension, spin axis | 중간 | 트랙맨이 측정하나 원자료 접근·정의 정합 필요 |
| plate_x/z 계열 | 낮음/주의 | KBO는 ABS(PTS) 기반이라 MLB 2026 ABS 변경과 별개로 좌표 정의가 다를 수 있음 |
| arm_angle | 낮음 | MLB도 2020+에만 있고, KBO 공개 여부 불명 |

→ 함의: 사용자의 전략("MLB에서 power 있는 feature를 만들고, KBO에 부족한 feature를 근거로 구단 설득")에서 **핵심 협상 재료는 release position/extension/spin axis 같은 중간 계층**이다. 구속·회전수는 이미 있으니 협상거리가 아니고, 이 중간 계층 feature가 MLB 모델 성능에 얼마나 기여하는지(feature ablation)를 보여주는 게 컨택의 실탄이 된다.

## 5. 적용 수준 — 단계별 현실 평가

- **지금 당장(외부 접근만으로): 사실상 불가.** 공개 pitch-level 데이터가 없어 KBO판 Kang 재현도, KBO reference set 구축도 불가능하다. 가능한 건 공개 집계 스탯(구속 평균 등) 수준의 매우 거친 근사뿐.
- **구단 1곳 협조 시: 부분 가능.** 그 구단 투수들의 트랙맨 원자료 + 내부 부상 기록을 받으면, MLB에서 학습한 distance metric/embedding을 그 구단 데이터에 적용하는 transfer가 가능. 단 positive(TJS) 수가 적어 그 구단만으로 검증은 어렵고, MLB 검증 결과에 의존해야 한다.
- **리그 차원 협조 시: 본격 가능.** 10구단 트래킹 원자료가 열리면 KBO 자체 reference set과 검증이 가능. 단 이건 정책·상업적 결정이라 프로젝트가 통제할 수 없는 변수.

## 5b. KBO 적용 후 검토 항목 (2026-07-13 사용자 결정 — 보류 목록)

- **불펜 별도 정렬 보조 명단**: 동결 모델은 불펜-내 판별력이 있으나
  (RP-내 ROC ~0.65/0.60) 순수 top-50 경보에서는 불펜 사건을 사실상
  놓친다 (0/37-40). RP 예약석 quota(q=20)는 safety 경계 게이트 실패로
  challenger 강등됨. **"불펜만 따로 정렬한 보조 명단" 방식은 KBO 적용
  후 검토 항목으로 보류** — KBO 사건 데이터에서 quota(q=5 포함)와 함께
  처음부터 재검정한다. 근거: results/phase3/A1_BULLPEN.md (정정 헤더).
- **제한형 연구 웹 대시보드 (사용자 추가 승인, 2026-07-13)**: 결정일별
  q=0 위험 순위 + 투수별 P90/P150·근거 분해 + "점수/순위 읽는 법" +
  검색·역할 필터·model/data hash를 제공한다. KBO 재적합을 기다리는 단순
  아이디어가 아니라, M1/M2 보수 뒤 MLB frozen archive와 retrospective
  demo만 읽는 로컬 one-command + 접근 제한형 인스턴스를 이번 연구에서
  먼저 **구현·배포**한다. 공식 제안서와는 독립이다. 필수 정직 요소:
  절대 확률 한계, sensitivity envelope, 불펜 coverage, snapshot 시점 표기.
  공개 실명 실시간 명단은 금지한다. 상세: plan_progress.md "계속 8".

## 6. 정직한 한계 (미확인 전제)

- 트랙맨 원자료에 실제로 어떤 컬럼이 포함되는지(release position/extension/spin axis의 존재·정밀도)는 구단 데이터를 직접 보기 전엔 확정 불가.
- KBO TJS 연간 건수의 정확한 수치는 공개 집계가 없어 이 문서에서 특정하지 못했다. "MLB보다 적다"는 리그 규모 기반 추정.
- ABS/PTS와 트랙맨의 좌표계 차이가 feature 정합에 주는 영향은 실데이터 비교 전엔 정량화 불가.

---

## 결론 요약

KBO 적용의 순서는 **(1) MLB에서 모델·feature 중요도 확립 → (2) 그 중 release position/extension/spin axis 등 "있으면 좋은데 KBO 공개가 막힌" feature의 기여도를 근거자료로 정리 → (3) 구단에 '데이터 접근'을 요청**이다. 병목은 feature 부족이 아니라 **데이터 공개 부재**이므로, 컨택 메시지는 "우리 MLB 모델이 이 feature들로 이만큼의 부상 신호를 잡는다. KBO 트랙맨에도 같은 feature가 측정되지만 접근이 안 된다. 데이터를 열어주면 KBO 투수에게 적용 가능하다"가 되어야 한다. 이는 MLB-first 전략이 KBO 적용의 전제 조건임을 다시 확인해준다.

---

## 2026-07-13 v2 정정 — 현재 canonical KBO 이전 판단

1. **공개 baseline은 조건부 가능**: 공식 KBO 화면에는 일자·등판 구분과
   박스스코어 투구수 필드가 존재한다. 다만 1군 장기 연속성·stable ID·
   bulk export/API와 연구용 저장·파생물 공개 권리는 아직 미확인이다.
   페이지가 보인다는 사실만으로 자동수집을 허가받은 것은 아니다. KBO/
   KBOP의 서면 허가 또는 승인된 export/API가 K1의 hard gate이며,
   STATIZ scraper는 서면 승인 전 사용하지 않는다.
   - 확인 근거: [KBO 선수 일별 기록](https://www.koreabaseball.com/Record/Player/PitcherDetail/Daily.aspx?playerId=65933),
     [공식 퓨처스 박스스코어 예시](https://www.koreabaseball.com/futures/schedule/BoxScore.aspx?gameId=20260507SSNC0&leagueId=2&seasonId=2026&seriesId=0),
     [KBO 이용약관](https://m.koreabaseball.com/Member/Join/Accessterms.aspx?appCk=false),
     [STATIZ 이용약관](https://www.statiz.co.kr/policy/?m=terms),
     [KBO 기록 시스템/데이터 문의 공지](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=11918).
2. **feature anchor는 B7/V9**: boxscore-only 동적 feature는 7개(B7)다.
   velocity 정의·단위·장기 coverage가 통과하면 vel_trend+vt_missing을
   더한 V9를 secondary로 연다. 실제 GS는 frozen start_share(50구 이상
   비율)의 대체가 아니라 역할 평가 변수다.
3. **primary 병목은 complete outcome ascertainment**: 뉴스/구단 발표는
   확정 양성 discovery에는 유용하지만 미보도자를 음성으로 둘 수 없는 PU
   자료다. 확인 사건 수는 사건 수의 하한일 뿐 ROC/성능의 하한이 아니다.
   공개 단계에서는 재적합·ROC/PR·보정 없이 확정 사례 percentile/lead를
   편향된 case-series로만 기술한다.
4. **내부 요청 우선순위**: (i) eligible roster 전체의 실제 UCLR/TJS
   시행일·술식·follow-up, (ii) 1·2군/재활 workload·roster·역할,
   (iii) IL 상세(auxiliary), (iv) tracking 증분 자료. exact surgery DB가
   이미 존재한다고 단정하지 않고 존재·형태부터 확인한다.
5. **학습·배포 분리**: complete internal labels를 확보한 뒤에만 KBO train
   scaler·계수·window calibration을 적합하고 untouched temporal test에
   1회 적용한다. MLB top-50/q20은 이식하지 않고 모집단 비율/구단당 용량
   기준을 재선택한다. 이후에도 사전 봉인 silent prospective를 통과하기
   전에는 월별 실명 명단이나 대시보드를 운영 배포하지 않는다.

따라서 현재 결론은 "tracking이 없어서 KBO 적용이 불가능"이 아니라,
**"허가된 boxscore B7 파이프라인은 선구축할 가치가 있지만, 정식 학습·
성능 검증의 병목은 완전한 TJS/UCLR 시행 라벨과 follow-up"**이다. 상세
실행 게이트는 `plan_progress.md`의 "2026-07-13 (계속 7)"이 우선한다.

**K0 실행 결과 (2026-07-14, 정본 = `results/kbo/K0_FEASIBILITY.md`)**:
판정 `PARTIAL`, K1 자동수집 `NO-GO`. 공식 boxscore에 gameId·경기일·
선발/구원·투구수는 존재하나 1군 다년 coverage·ID 연속성·risk-set replay
미확인, 공식 export/API·연구 bulk 허가 부재(약관 사전승낙 범위), 공개
structured daily velocity 부재로 V9 공개 경로 FAIL. **후속 사용자
결정**: 과거 KBO 홈페이지 수집 경험 + 네이버 문자중계 PTS 구속 경로
제시, 수집 주체는 Codex. 다음 후보 = label-blind 소규모 source/schema/
coverage pilot (KBO 공식 투구수 = B7 정본, PTS 구속 = 별도 `V9-PTS`
후보; 별도 승인 후 착수). PU 원칙(`unknown != negative`, 공개 자료
재적합·ROC 금지)은 불변 — plan_progress "2026-07-14 (계속 2)".

---

## 2026-07-14 K0 실행 판정 — v2 gate 적용 결과

위 v2의 K0를 공식 KBO 1차 자료에 한해 읽기 전용으로 실행했다. 상세 정본은
`results/kbo/K0_FEASIBILITY.md`이며, 이 절이 K0의 현재 상태다.

1. **K0=`PARTIAL`, K1 자동수집=`NO-GO`.** 퓨처스 실제 박스스코어 여러
   연도 표본에서 `gameId`, 경기일, 선발/구원, 투수별 투구수를 확인했다.
   1군 player daily에는 역할·날짜는 있지만 경기별 투구수가 없고, 1군
   GameCenter의 채워진 다년 표본·coverage·ID 영속성·historical risk-set
   replay는 확인하지 못했다.
2. **권리 gate=`FAIL`.** 공식 export/API 또는 연구 목적의 자동수집·장기
   저장·snapshot·파생물 공유 허가를 확인하지 못했다. KBO 약관은 서비스
   정보의 개인적 이용 외 복제·제3자 제공 등에 사전승낙 범위를 둔다.
   페이지 공개는 bulk 권리가 아니므로 scraper·비공식 endpoint·STATIZ
   우회 수집을 시작하지 않는다.
3. **B7=`PARTIAL`, V9 공개 경로=`FAIL`.** 실제 PC 기반 B7은 기술적으로
   유망하지만 권리와 1군 장기 완전성이 선행되어야 한다. KBO가 2025년부터
   공식 구속을 TrackMan으로 일원화한 사실은 PASS이나, 공개 structured
   daily velocity가 없어 vel_trend/vt_missing을 계산할 수 없다.
4. **공개 outcome은 PU.** KBO 도메인 기사로 A/B/C evidence registry는
   만들 수 있으나 미발견은 negative가 아니다. public-only 단계에서
   refit·prevalence·ROC/PR·calibration은 계속 금지한다.
5. **다음 순서:** 정당한 권리자에게 export/API 또는 서면 연구 허가를
   확보한 뒤, 소수의 1군+퓨처스 경기와 예외 경기만으로 schema/completeness
   pilot을 먼저 수행한다. 전수 수집은 그 pilot의 별도 go 이후다. 현재는
   B7 설계만 유지하고 K1을 열지 않는다.

MLB retrospective D0 dashboard는 별도 트랙으로 구현·production 검증을
마쳤다. KBO 실명 운영 화면으로 확장한 것이 아니며, 사용자 source
commit/push 뒤 owner-only Sites publish만 남았다.
