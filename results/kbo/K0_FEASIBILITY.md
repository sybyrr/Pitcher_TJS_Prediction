# K0 — KBO 공개 데이터 이전 가능성·권리 게이트

- 조사일/접근일: **2026-07-14 (KST)**
- 범위: KBO 공식 페이지와 KBO 도메인에 게시된 사례 자료의 **읽기 전용 표본 확인**
- 수행하지 않은 것: 외부 연락, 대량 다운로드, 자동 수집, 비공개 API 호출, 데이터셋 생성, 코드 실행·수정
- 내부 기준: KBO-B7/V9, TJS-only, 공개 사례의 `unknown != negative`, A/B/C evidence ontology

## 1. 결론

**K0 종합 판정은 `PARTIAL`, K1 자동 수집은 `NO-GO`다.**

공식 웹 화면에는 KBO-B7을 구성할 가능성이 있는 경기일·선발/구원 구분·투수별 투구수와 `playerId`/`gameId` 후보가 보인다. 특히 퓨처스리그 박스스코어에서는 여러 연도의 실제 경기 표본으로 이 조합을 확인했다. 따라서 “KBO 공개 화면에는 필요한 원자료가 전혀 없다”는 결론은 옳지 않다.

그러나 다음 두 가지 하드 블로커가 남는다.

1. **권리:** 공개된 공식 export/API 또는 연구 목적 자동 수집·저장·재배포 허가를 확인하지 못했다. 오히려 KBO 이용약관은 개인적 이용 범위를 벗어난 복제와 제3자 제공 등에 사전 동의를 요구한다. 공개 열람 가능성은 bulk 수집 권리가 아니다.
2. **완전성:** 1군 경기별 투구수 화면의 실제 장기 표본과 연도별 완전성, 과거 등록/말소·선수 이동 이력의 재현 가능성, ID의 공식 안정성 계약을 확인하지 못했다.

따라서 현재 증거는 **수동 schema feasibility 확인**에는 충분하지만, K1에서 자동 수집을 시작하거나 분석용 종단 데이터셋을 구축할 근거로는 부족하다. 권리 게이트와 1군 장기 coverage 게이트가 모두 해소될 때까지 B7은 설계 상태로 유지하고, V9는 공개 데이터 경로에서 닫아 두는 것이 타당하다.

## 2. 판정 기준

- `PASS`: 공식 1차 자료에서 요구 항목과 필요한 범위를 직접 확인했고, 다음 단계 진행을 막는 핵심 불확실성이 없음.
- `PARTIAL`: 표본 또는 화면 schema는 확인했으나 장기 완전성·식별자 안정성·재현성 중 하나 이상이 미확인.
- `FAIL`: 필수 자료 또는 사용 권리를 확인하지 못했거나, 확인된 조건이 현재 계획의 실행을 막음.

| 게이트 | 판정 | 핵심 근거 | K1 영향 |
|---|---:|---|---|
| 공식 export/API 또는 서면 허가 | **FAIL** | 공개 문서를 찾지 못했고, 약관은 사전 동의가 필요한 이용을 명시 | **자동 수집 하드 중단** |
| 1군 경기별 투구수·역할·날짜 | **PARTIAL** | 공식 경기 리뷰 표에 필드는 있으나 실제 다년 표본/coverage 미확인 | B7 자동 구축 불가 |
| 퓨처스 경기별 투구수·역할·날짜 | **PARTIAL** | 2015·2016·2019·2024·2026 실제 표본 확인; 연속·전수 coverage 미확인 | 보조 workload 후보만 유지 |
| `playerId`/`gameId`와 팀 이동 연결 | **PARTIAL** | URL key와 선수 이력은 확인; 공식 영속성·중복/병합 규칙 미확인 | 종단 join 금지 |
| 과거 등록/말소·이적·1군/퓨처스 risk set | **PARTIAL** | 현재 일자와 transaction schema는 확인; 역사 replay와 완전성 미확인 | censoring/분모 구성 불가 |
| B7 공개 자료 기술 가능성 | **PARTIAL** | 필요한 값의 상당 부분은 화면에 있으나 1군 핵심 PC history와 권리가 미해결 | 설계 유지, 실행 보류 |
| 2025 이후 구속 측정 표준화 | **PASS** | KBO가 TrackMan 일원화를 공식 발표 | 측정 체계 사실만 PASS |
| V9용 공개 일자별 구속 시계열 | **FAIL** | 공식 daily/boxscore 표에 구속 필드가 없고 구조화된 공개 export를 못 찾음 | V9 공개 경로 폐쇄 |
| 공개 TJS/UCLR case registry | **PARTIAL** | A/B/C 사례 분류는 가능하나 완전한 positive ascertainment와 negatives가 없음 | PU 기술만 가능 |
| 공개 자료만으로 감독학습/성능평가 label | **FAIL** | 미보도·미확인 선수를 음성으로 둘 수 없음 | refit, ROC/PR, calibration 금지 |

## 3. 공식 기록 화면에서 확인한 schema와 coverage

### 3.1 1군 선수·경기 기록

#### 선수 일자별 기록

[KBO 투수 일자별 기록 표본 (`playerId=52868`)](https://www.koreabaseball.com/Record/Player/PitcherDetail/Daily.aspx?playerId=52868)에서 다음을 확인했다.

- URL에 `playerId`가 존재한다.
- 연도 선택 UI는 2001~2026을 제시한다.
- 표에는 경기일, 상대, `구분`(선발/구원), TBF, 이닝, 피안타, 볼넷, 탈삼진, 실점 등이 있다.
- **경기별 투구수와 구속은 이 표에 없다.**

연도 선택 UI는 “해당 연도의 모든 투수·경기 행이 완전하고 같은 schema로 제공된다”는 증거가 아니다. 이번 K0에서는 한 선수의 현재 실제 행만 읽었으며, 모든 연도를 열거하거나 내려받지 않았다.

#### 경기 리뷰의 투수 표

[KBO 1군 경기 리뷰](https://www.koreabaseball.com/Schedule/GameCenter/ReviewNew.aspx)의 투수 표 schema에는 선수, 등판, 결과, 이닝, 타자, **투구수**, 탈삼진, 실점 등이 보인다. `등판` 필드는 선발 또는 구원 등판 위치를 표현할 수 있는 구조다.

다만 이 화면은 동적으로 경기 상태를 불러온다. 이번 읽기 전용 조사에서는 특정 과거 경기의 채워진 1군 표본과 URL의 안정적인 `gameId`를 재현하지 못했다. 따라서 “1군에도 투구수 필드가 있다”는 schema 사실과 “계획 기간 전체의 경기별 투구수를 안정적으로 수집할 수 있다”는 coverage 주장을 분리해야 한다.

[KBO 경기 일정](https://www.koreabaseball.com/schedule/schedule.aspx)도 날짜·경기·구장·GameCenter 열을 제공하지만, 정적 응답만으로 과거 전수 경기와 GameCenter key 연결을 검증하지 못했다.

#### 선수 ID와 팀 이동

[KBO 투수 기본 기록 표본 (`playerId=60768`)](https://www.koreabaseball.com/Record/Player/PitcherDetail/Basic.aspx?playerId=60768)은 한 선수 페이지 안에서 여러 구단 소속 시즌 이력을 보여 준다. 같은 URL key 아래 과거 팀 이력이 나타나므로 `playerId`는 종단 연결의 유력한 후보다. 또한 해당 페이지의 시즌 합계 표에는 `NP`(투구수)가 있지만, 최근 경기 표에는 경기별 `NP`가 없다.

[장기 선수 시즌 기록 표본 (`playerId=75340`)](https://www.koreabaseball.com/record/Player/PitcherDetail/Total.aspx?playerId=75340)에서는 2000년대부터 2020년대까지 실제 시즌 합계가 나타난다. 이는 오래된 선수 이력이 남아 있다는 표본 증거이지, 일자별/경기별 행의 같은 기간 coverage 증거는 아니다.

확인하지 못한 항목은 다음과 같다.

- `playerId`의 공식 불변성 보장, 개명·동명이인·재등록·외국인 재입단 처리
- 1군과 퓨처스 페이지에서 동일 ID가 항상 유지되는지
- `gameId` 형식과 영속성을 보장하는 공개 명세
- 취소·서스펜디드·더블헤더·재개 경기의 key와 날짜 규칙

### 3.2 퓨처스리그 실제 박스스코어

다음 공식 경기 표본은 URL에 `gameId`, `leagueId`, `seasonId`, `seriesId`를 가지며, 화면에 경기일과 투수별 `등판`, 이닝, 타자, **투구수**를 표시한다. 선발 투수는 `등판` 열에서 선발로 구분된다.

- [2015-06-03 표본](https://www.koreabaseball.com/Futures/Schedule/BoxScore.aspx?gameId=20150603SKSM0&leagueId=2&seasonId=2015&seriesId=0)
- [2016-08-07 표본](https://www.koreabaseball.com/Futures/Schedule/BoxScore.aspx?gameId=20160807SKSM0&leagueId=2&seasonId=2016&seriesId=0)
- [2019-09-19 표본](https://www.koreabaseball.com/futures/schedule/BoxScore.aspx?gameId=20190919SKKT0&leagueId=2&seasonId=2019&seriesId=0)
- [2024-08-13 표본](https://www.koreabaseball.com/futures/schedule/BoxScore.aspx?gameId=20240813SKOB0&leagueId=2&seasonId=2024&seriesId=0)
- [2026-05-06 표본](https://www.koreabaseball.com/Futures/Schedule/BoxScore.aspx?gameId=20260506ULOB0&leagueId=2&seasonId=2026&seriesId=0)

[KBO 퓨처스 투수 통산기록](https://www.koreabaseball.com/Futures/Player/PitcherTotal.aspx?playerId=68341)은 화면 제목에서 통산기록 범위를 “2010년 이후”로 설명한다. 그러나 이는 시즌 합계 페이지의 범위이며, 모든 2010년 이후 경기 박스스코어가 전수 보존되었다는 뜻은 아니다.

따라서 퓨처스 경기 PC는 **기술적으로 유망하지만 `PARTIAL`**이다. 서로 떨어진 다섯 연도의 실제 표본은 확인했지만, 연속 시즌 전수성, 우천 취소·재편성, 비공식 교육리그/재활 경기, 선수 ID join, 정정 반영 방식은 검증하지 않았다.

이 자료는 불펜·재활 workload 누락을 줄일 가능성이 있다. 다만 현재 canonical B7의 1군 risk set을 조용히 1·2군 혼합 모델로 바꾸면 안 된다. 권리와 완전성이 확보된 뒤에도 퓨처스 PC는 먼저 다음 중 하나로 명시해야 한다.

1. 1군 예측시점 이전의 보조 workload history
2. 2군 등판을 포함하는 별도 sensitivity
3. 내부 데이터와의 누락 점검용 audit field

### 3.3 등록·말소, 1군/퓨처스, 이적

[KBO 1군 선수 등록 현황](https://www.koreabaseball.com/Player/Register.aspx)은 조회 일자의 구단별 등록선수와 당일 등록·말소 구역을 제공한다. [전체 등록·말소 화면](https://www.koreabaseball.com/Player/RegisterAll.aspx)과 [퓨처스 등록 현황](https://www.koreabaseball.com/Futures/Player/Register.aspx)도 현재 상태를 분리해 보여 준다.

[KBO 선수 이동 현황](https://www.koreabaseball.com/Player/Trade.aspx)은 일자, 구분, 구단, 선수, 비고 schema와 함께 트레이드, 계약해지, FA, 임의해지, 부상자명단, 군보류 등 여러 category 선택지를 제공한다.

이번 표본 확인으로는 다음을 입증하지 못했다.

- 과거 임의 일자의 전체 등록명단을 동일 화면에서 재생할 수 있는지
- daily 등록/말소와 이동 기록이 계획 기간에 대해 누락 없이 공개되는지
- 거래/등록 행에 안정적인 `playerId`가 노출되는지
- 퓨처스 이동과 재활 등판까지 합쳐 eligibility·censoring을 재구성할 수 있는지

즉, 현재 화면은 오늘의 roster 확인과 사건 schema 이해에는 쓸 수 있지만, **역사적 risk set의 분모를 만드는 증거로는 부족하다.**

KBO는 [2020년 부상자명단 제도 도입 공지](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=7651)에서 10·15·30일 단위 신청과 진단서 제출 절차를 설명한다. 이는 구단/KBO 내부에 의료 근거가 존재할 수 있음을 보여 주지만, 공개 등록 화면이 진단명이나 수술 여부를 제공한다는 뜻은 아니다. 따라서 IL 표시는 TJS label도, TJS가 아니라는 negative label도 아니다.

### 3.4 기록 정정과 경기 예외

[KBO 기록 정정 현황](https://www.koreabaseball.com/record/recordcorrect/recordcorrect.aspx)은 경기일, 경기, 이닝, 최초 기록, 정정 기록, 내용, 정정일 schema를 공개한다. 데이터 접근이 허가되더라도 원자료 snapshot 날짜, 정정 적용일, 재수집 정책을 반드시 남겨야 한다.

[2024 더블헤더 운영 공지](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=9859)와 [2024 한국시리즈 서스펜디드 경기 공지](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=10269)는 날짜 문자열만으로 경기 단위를 추정하면 안 되는 실제 예외가 있음을 보여 준다. `gameId`의 공식 규칙을 얻기 전에는 ID를 파싱해 날짜·팀·경기번호를 재구성하지 말고 opaque key로 다뤄야 한다.

## 4. B7/V9와의 직접 대조

### 4.1 KBO-B7

| B7 변수 | 필요한 원자료 | 공개 화면 확인 | 판정/주의 |
|---|---|---|---|
| `pc_chronic` | 연속 경기별 투구수·날짜 | 퓨처스 actual PASS, 1군 schema만 확인 | **PARTIAL** |
| `pc_acute_dev` | 최근 PC와 과거 workload 기준 | 위와 동일 | **PARTIAL** |
| `days_since_last` | 선수별 등판일 | 1군 daily와 퓨처스 boxscore에 존재 | schema **PASS**, 종단 join은 PARTIAL |
| `month` | 경기일 | 존재 | **PASS** |
| `start_share` | 경기별 PC로 `50+ pitch game` 비율 계산 | PC가 있으면 계산 가능 | 공식 GS를 대체값으로 쓰지 않음 |
| `prior_pc_rate` | 과거 경기별 PC와 관찰시간 | 다년 완전성 미확인 | **PARTIAL** |
| `ncg_log` | 누적 eligible game count | ID·전수 경기·risk set 필요 | **PARTIAL** |

핵심은 `start_share`다. 공식 화면의 선발/구원 표시는 역할 타당도 확인과 sensitivity에는 유용하지만, canonical 정의인 **50구 이상 경기 비율을 GS 비율로 바꾸면 MLB 동결 모델과 다른 feature가 된다.** 불펜 투수도 실제 투구수 행을 포함해 동일한 PC 기반 규칙으로 계산해야 한다.

현재 공개 화면 구조만 보면 불펜 workload를 포함한 B7은 가능성이 있다. 하지만 1군 경기별 PC 장기 coverage와 허가가 없으므로 계산을 시작할 수는 없다.

### 4.2 KBO-V9와 TrackMan

KBO의 [2025 공식 구속 측정 TrackMan 일원화 공지](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=11398)는 다음 경계가 있음을 확인한다.

- 2025년부터 공식 구속 측정을 TrackMan으로 일원화한다.
- 이전에는 방송사·구장별 측정 방식 차이가 있었다.
- 2024년에는 일부 구장 전광판만 TrackMan 값을 사용했고, 나머지는 2025년부터 순차 전환 대상이었다.

[2025시즌 제도 변경 안내](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=11416)도 전 구장 TrackMan 기반의 통일된 구속 표시를 설명한다. 이 공지는 **2025 이후 측정 체계의 비교 가능성이 이전보다 좋아졌다는 근거**다.

그러나 확인한 공식 선수 daily, 1군 경기 리뷰, 퓨처스 boxscore에는 일자별 평균/최고 구속 열이 없다. KBO 도메인의 개별 기사에서 특정 경기 평균·최고 구속을 언급하는 경우는 있지만 선택적으로 보도된 서술형 값이며, 전수 시계열·측정 정의·결측 사유가 보장되지 않는다. 이를 수집해 V9의 `vel_trend`로 만들면 보도 선택 편향이 생긴다.

따라서:

- **2024 이전:** 측정원 혼합 때문에 같은 의미의 장기 trend로 합치지 않는다.
- **2025 이후:** 측정 표준화 사실은 PASS지만, 공개 structured daily values는 FAIL이다.
- **V9:** 공식 export/API 또는 허가된 TrackMan 자료가 생기기 전까지 공개 경로에서 **닫음**.
- `vt_missing`은 부분 결측을 표현하는 장치이지, 관측 구속 자체가 거의 없는 상황을 정당화하는 장치가 아니다.

## 5. 공개 TJS/UCLR registry의 가능성과 PU 한계

### 5.1 A/B/C ontology 적용 가능성

KBO 도메인에 게시된 기사만으로도 evidence grade를 구분해야 하는 실제 사례가 있다.

- [안우진 수술 시행 기사](https://www.koreabaseball.com/MediaNews/News/KboPhoto/View.aspx?bdSe=427825): 특정 일자에 UCL reconstruction을 **받았다고 사후 확인**한다. procedure가 명확하므로 A 후보다.
- [주승우 수술 예정 기사](https://www.koreabaseball.com/MediaNews/News/KboPhoto/View.aspx?bdSe=501834): 구단이 특정 미래 날짜의 UCL reconstruction 계획을 발표한다. 날짜가 정확해도 시행 전에는 **C**다.
- [윤영철 수술 예정 기사](https://www.koreabaseball.com/MediaNews/News/KboPhoto/View.aspx?bdSe=501996): 특정 수술 예정일 보도 역시 시행 확인 전에는 **C**다.
- [이의리 수술 계획 기사](https://www.koreabaseball.com/MediaNews/News/KboPhoto/View.aspx?bdSe=445739): UCL reconstruction 계획은 있으나 당시 날짜가 미정인 사례로, **C**다.

이 페이지들은 KBO의 공식 경기·등록 DB가 아니라 KBO 도메인에 재게시된 미디어 기사다. 따라서 도메인만 보고 official medical registry로 승격시키면 안 되며, 기사 원출처·구단 발표 인용 여부·사후 시행 확인을 별도 필드로 남겨야 한다.

canonical ontology 적용은 다음과 같다.

- **A:** UCLR/Tommy John을 실제로 시행했다는 확인과 정확한 수술일이 모두 있음.
- **B:** 실제 시행은 확인되지만 수술일이 구간으로만 특정됨.
- **C:** 수술 예정·권고·검토 단계이거나, 팔꿈치 수술이라는 사실만 있고 UCLR인지 불명확함.

예정일은 실제 수술일이 아니다. C 사건은 사후 출처가 생길 때만 A/B로 승격한다. 팔꿈치 통증, UCL 손상, IL 등록, 재활, 단순 수술 언급도 TJS-only positive로 자동 변환하지 않는다.

### 5.2 왜 public registry는 PU인가

공개 기사 검색으로 얻는 것은 “보도되고 발견된 positive”다. 다음을 알 수 없다.

- 모든 KBO/퓨처스 투수의 수술 여부가 동일 확률로 보도·색인되었는지
- 해외·아마추어·입단 전 수술과 재수술이 빠지지 않았는지
- 수술을 받지 않았다는 확인이 있는 진짜 negative가 누구인지
- 발표일과 실제 수술일 사이의 누락·정정이 얼마나 있는지

그러므로 공개 자료에서 미발견된 선수/기간은 **negative가 아니라 unknown**이다. 공개 registry만으로는 다음을 하면 안 된다.

- KBO prevalence 추정
- ROC-AUC, PR-AUC, sensitivity/specificity 계산
- Brier score와 calibration 평가
- 모델 재학습·재보정
- 미보도 선수를 0으로 둔 case-control 분석

허용 가능한 public-only 산출은 A/B 사건에 대한 score percentile, event 전 lead-time, evidence grade별 사례 기술처럼 negative를 요구하지 않는 제한적 점검이다. 이것도 먼저 유효한 workload 데이터와 시점 정렬이 있어야 한다. 감독학습과 정식 성능 평가는 구단/리그 등에서 완전한 TJS-only ascertainment를 제공받을 때만 가능하다.

## 6. 이용권리·저장·재배포 게이트

[KBO 서비스 이용약관](https://m.koreabaseball.com/Member/Join/Accessterms.aspx?appCk=false)은 2024-02-26 시행 약관으로 표시되며, 서비스 정보의 개인적 이용 범위를 벗어난 복제·게시·방송·제3자 제공 등에 사전 동의가 필요한 조건을 둔다. 제휴 게시물과 콘텐츠의 이용·가공·판매에도 별도 권리 제한이 있다.

공식 사이트에서 API, CSV/Excel export, 비상업 연구용 bulk access, 원자료의 로컬 저장, snapshot 보존, 파생 데이터 재배포를 명시적으로 허용하는 공개 문서는 이번 조사에서 찾지 못했다. [KBO 기록위원회 게시판](https://www.koreabaseball.com/Kbo/AboutKbo/Committee/ScoringList.aspx)에는 API·상업적 이용을 문의하는 사용자 글 제목이 보이지만 공개 답변이나 사용 허가 문서가 아니며, 1:1 문의 안내를 허가로 해석할 수 없다.

검색으로 공개 문서를 찾지 못했다는 사실은 법적으로 “그런 계약이 존재하지 않는다”는 증명은 아니다. 다만 연구 운영상 다음 결론은 명확하다.

> **공개 웹페이지를 볼 수 있다는 사실은 자동 수집, 장기 저장, 파생 데이터 공개 또는 재배포 권리가 아니다.**

이 문서는 법률 자문이 아니라 프로젝트의 보수적 데이터 거버넌스 판정이다. 현재 상태에서 페이지를 순회하는 scraper, 비공개 endpoint 역공학, 대량 요청, 원문/표의 저장을 시작하지 않는다.

## 7. K1 자동 수집 Go/No-Go

### 현재 결정

**K1 자동 수집: `NO-GO`.**

`NO-GO`의 직접 원인은 다음과 같다.

1. 공식 export/API 또는 서면 허가가 없음.
2. 약관상 사전 동의가 필요한 이용 범위와 충돌할 가능성을 해소하지 못함.
3. 1군 경기별 투구수의 실제 다년 coverage와 안정적인 game join을 검증하지 못함.
4. 과거 roster·등록/말소·이적 history로 risk set과 censoring을 재구성할 수 있는지 불명확함.
5. V9용 구조화된 일자별 구속값이 없음.
6. public TJS 사례는 PU라 supervised label이 될 수 없음.

### K1을 열기 위한 필요조건

아래는 **모두 충족해야 하는 순차 게이트**다.

1. KBO, Sports2i 또는 정당한 데이터 권리자로부터 공식 export/API 혹은 서면 허가를 확보한다.
2. 허가 문서에서 비상업 연구 목적, 자동 요청 범위, rate limit, 로컬 저장, snapshot/backup, 보유·삭제 기간, 파생 feature와 집계 결과의 공유 가능 범위를 확인한다.
3. 허가된 방식으로만 소수 경기 pilot을 수행해 1군과 퓨처스의 `playerId`, `gameId`, 날짜, 투수별 PC, 역할, 팀, 리그 level schema를 검증한다.
4. 서로 다른 과거 시즌, 더블헤더, 서스펜디드/재개, 정정 경기, 트레이드·개명·재입단 선수 표본을 포함해 ID와 시간 규칙을 검증한다.
5. 경기 행을 시즌 합계 G/NP와 대조해 누락·중복을 정량화하고, 허용 기준을 사전에 고정한다.
6. 과거 등록/말소와 선수 이동을 replay할 수 없으면 1군 등판 기반의 제한된 risk set으로 연구 질문을 축소하고 그 한계를 명시한다.
7. V9는 별도 TrackMan export에서 일자별 값, 단위, pitch type 집계법, 측정원, 2025 전환 시점, 결측 사유가 확인될 때만 재개한다. 그렇지 않으면 B7-only를 유지한다.
8. TJS/UCLR outcome은 완전한 내부 ascertainment를 받지 않는 한 public PU 사례 점검으로만 유지한다.

권리 게이트가 PASS가 되더라도 곧바로 전기간 수집으로 넘어가면 안 된다. 먼저 위의 작은 schema/completeness pilot을 통과시키고, 그 결과를 보고 K1 본 수집 여부를 다시 승인받는 것이 적절하다.

## 8. 최종 권고

| 선택지 | 장점 | 치명적 한계 | 권고 |
|---|---|---|---:|
| 지금 공개 페이지 자동 수집 | 빠르게 보이는 데이터를 모을 수 있음 | 권리 FAIL, 1군 coverage·ID 불명, 재현성과 배포 위험 | **기각** |
| 권리 확인 후 B7 pilot | 실제 PC 기반 불펜 포함 workload와 MLB frozen score의 transport 점검 가능 | 허가와 schema 검증이 선행돼야 함 | **1순위** |
| 공개 기사로 TJS 0/1 label 생성 | 표면상 데이터셋을 빨리 만들 수 있음 | unknown을 negative로 오인하는 중대한 bias | **기각** |
| public A/B case registry만 구축 | 사례 percentile·lead-time 기술 가능 | PU이며 성능평가·재학습 불가 | **권리 확인 후 제한적으로** |
| 2025+ 공개 기사 구속으로 V9 | 일부 사례의 구속을 얻을 수 있음 | 선택 보도, 정의·결측·전수성 없음 | **기각** |
| 허가된 TrackMan export로 2025+ V9 | 측정 체계가 비교적 일관됨 | 짧은 역사와 2024 이전 단절 | **자료 확보 시 별도 재심사** |

현재 연구의 가장 정당한 다음 상태는 **“KBO 이전 포기”가 아니라 “B7은 기술적으로 유망하나 권리·완전성 gate 대기, V9는 공개 경로에서 보류, outcome은 PU로 제한”**이다. 특히 불펜 관점은 1군 경기별 실제 투구수 전수 자료가 허가되면 살릴 수 있지만, 공식 GS 비율로 `start_share`를 대체하거나 퓨처스/기사 자료를 검증 없이 합쳐서는 안 된다.

## 9. 출처 성격과 한계

- 기록·일정·등록·약관·제도 공지는 KBO 공식 도메인의 1차 자료로 취급했다.
- KBO 미디어 페이지의 수술 기사는 KBO 도메인에 있어도 경기 기록 DB가 아니며, 기사 원출처와 구단 발표 인용을 확인해야 하는 사례 근거다.
- 표본 확인은 대표 페이지를 수동으로 읽은 것이며 전수 crawl이나 endpoint 호출 결과가 아니다.
- 동적 1군 GameCenter의 채워진 과거 경기 표본을 이번 환경에서 재현하지 못했다. 따라서 해당 부분을 PASS로 올리지 않았다.
- 모든 웹 출처의 마지막 접근일은 **2026-07-14**다.

