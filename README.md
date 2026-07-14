# Pitcher TJS Prediction

MLB 투수의 Tommy John surgery(TJS) 위험 신호를 전향적으로 계산하고,
검증된 정의와 평가 절차를 KBO 환경으로 이전하기 위한 연구 저장소다.

이 프로젝트는 Kang et al. (2025) 코드의 충실 재현에서 시작했지만, 현재
정본은 회고적 딥러닝 분류기가 아니라 **의사결정 시점 이전 정보만 사용하는
90일·150일 TJS 위험 순위 모델**이다. MLB 모델은 2026-07-13에 동결했으며,
KBO에서는 정의와 절차만 이전하고 계수·보정·경보 용량은 다시 추정한다.

## 현재 상태

| 영역 | 상태 | 정본 |
|---|---|---|
| Kang 재현·해체 | 완료 | [`phase2_results.md`](phase2_results.md) |
| MLB 전향 위험 모델 | 동결 완료 | [`results/phase3/FROZEN_MODEL.md`](results/phase3/FROZEN_MODEL.md) |
| 감사 보수 A0/A1/A-IL | 완료 | [`results/phase3/`](results/phase3/) |
| fit-free 2026 채점 경로 | 테스트 9/9 통과 | [`results/phase3/M2_FROZEN_SCORER.md`](results/phase3/M2_FROZEN_SCORER.md) |
| 연구 대시보드 | 로컬 build/test 완료, 원격 배포 보류 | [`dashboard/README.md`](dashboard/README.md) |
| KBO 이전 | 공개 소스 K0 감사 완료, 수집 pilot 전 | [`results/kbo/K0_FEASIBILITY.md`](results/kbo/K0_FEASIBILITY.md) |

다음 KBO 실행 블록은 결과를 보지 않는 source/schema/coverage pilot이다.
현재는 팀 인계를 위한 저장소 정비까지만 완료했으며 KBO 데이터 수집은 아직
시작하지 않았다.

MLB 대표 결과는 H90 ROC **0.701 [0.643, 0.759]**, H150 ROC
**0.696 [0.645, 0.746]**이다. 이는 untouched test가 아니라 adaptive
selection 이후의 **조건부 backtest**다. 더 보수적인 2-year safety
경계에서는 0.660/0.665로 약해진다. 절대확률은 validation 재보정 전 인용하지
않으며, 자세한 인용 규칙과 철회 수치는 [`PROJECT_MEMORY.md`](PROJECT_MEMORY.md)에
있다.

첫 진짜 전향 cohort는 **2026-08-01 당일 또는 이전에** 최신 원자료로 점수를
계산하고 snapshot·hash를 봉인해야 한다. 2026년 4–7월 archive는 결정일
이후에 복원한 delayed shadow backfill이므로 prospective 결과가 아니다.

## 처음 읽는 순서

사람과 에이전트 모두 아래 순서를 권장한다.

1. [`README.md`](README.md) — 프로젝트 목적, 구조, 현재 상태
2. [`PROJECT_MEMORY.md`](PROJECT_MEMORY.md) — 작업 원칙, canonical 수치,
   금지된 주장, 기각 이력, 데이터 지도
3. [`plan_progress.md`](plan_progress.md) — 전체 계획과 세션별 결정 로그;
   **파일 끝의 최신 절이 현재 상태**
4. [`progress_MLB.md`](progress_MLB.md) — MLB 연구의 일반 독자용 흐름
5. [`phase2_results.md`](phase2_results.md) — 재현·진단·전향 재설계 결과
6. [`results/phase3/FROZEN_MODEL.md`](results/phase3/FROZEN_MODEL.md) —
   동결 모델과 평가 프로토콜의 단일 원본
7. [`results/kbo/K0_FEASIBILITY.md`](results/kbo/K0_FEASIBILITY.md)와
   [`KBO_applicability.md`](KBO_applicability.md) — KBO source/label/이전 경계

에이전트별 진입점은 [`AGENTS.md`](AGENTS.md)와
[`.claude/CLAUDE.md`](.claude/CLAUDE.md)다. 두 파일보다
`PROJECT_MEMORY.md`의 내용이 우선한다.

## 연구 흐름

```text
Kang 코드·데이터 재현
  -> 누수·fill artifact·split 문제 진단
  -> as-of-date 전향 태스크 재정의
  -> MLB 9-feature hazard model 검증·동결
  -> fit-free prospective scorer와 제한형 대시보드
  -> KBO workload/PTS/TJS source pilot
  -> KBO 데이터가 충분할 때만 재적합·locked temporal test
  -> silent prospective 후 제한적 내부 pilot
```

KBO 공개 기사에서 발견한 수술 사례는 positive-unlabeled(PU) 자료다.
미발견 선수를 음성으로 바꾸지 않으며, complete outcome 확인 전에는 KBO
재학습·ROC/PR·보정·유병률 주장을 하지 않는다. 라벨은 연구 전 과정에서
**TJS/UCLR surgery only**로 유지하고 일반 IL을 섞지 않는다.

## 저장소 지도

| 경로 | 역할 | Git |
|---|---|---:|
| `src/` | 초기 재현·Phase 2 학습 코드 | 추적 |
| `scripts/` | 환경 확인과 공용 도구 | 추적 |
| `results/phase2/` | Phase 2 결정 CSV·요약 | 추적 |
| `results/phase3/` | 전향 모델 실험, 동결 state, scorer, 테스트 | 추적 |
| `results/kbo/` | KBO feasibility와 이후 결정 메모 | 추적 |
| `dashboard/` | retrospective 연구 대시보드 | 추적 |
| `data/` | Statcast, cohort, label snapshot, IL 원자료·파생표 | **Git 제외** |
| `TJS_Prediction/` | 수정 금지 upstream clone과 저자 제공 데이터 | **Git 제외** |

`results/phase3/frozen_model_state.json`은 full-precision 동결 상태로 Git에
포함된다. 반면 raw/derived data와 저자 제공 `final_df.csv`는 포함되지 않는다.
외부 데이터는 상대경로를 보존한 제한형 Drive 패키지와 파일별 hash manifest로
전달한다.

## 환경 구성

- Windows 11 / PowerShell
- Python 3.11.11, 프로젝트 로컬 `.venv`
- PyTorch CUDA 12.8 계열
- 대시보드: Node.js 22.13 이상

Python 환경은 저장소의 lock 파일을 사용한다.

```powershell
git clone https://github.com/dxlabskku/TJS_Prediction.git
cd TJS_Prediction
git checkout 6c71573fcc748e18a4522036575925db7e091e4b
cd ..

uv venv .venv --python 3.11.11
uv pip sync requirements.lock.txt --python .venv\Scripts\python.exe --torch-backend cu128
.venv\Scripts\python.exe scripts\verify_env.py
```

환경 검증 기대값은 **9/9 PASS**다. 이 전체 검증은 NVIDIA CUDA GPU,
pretrained weight 및 Statcast 표본을 받을 인터넷 연결, pybaseball cache
쓰기 권한을 요구한다. 해당 조건이 없는 CPU·offline 환경의 실패를 코드
회귀로 해석하지 않는다. Statcast 대량 수집 시에는
`pybaseball.cache.enable()`을 사용한다.

대시보드 실행:

```powershell
cd dashboard
npm ci
npm test
npm run dev
```

Windows PowerShell 실행 정책으로 `npm.ps1`이 차단되면 `npm.cmd`를 사용한다.
대시보드는 versioned CSV/manifest를 읽을 뿐 모델을 학습하거나 데이터를
다운로드하지 않는다.

## 데이터가 필요한 작업

GitHub clone만으로 문서·결과·동결 계수는 검토할 수 있다. 다음 작업에는
별도 데이터 패키지가 필요하다.

| 목적 | 필요한 외부 데이터 |
|---|---|
| 현재 Phase 3 결과 재검산 | `data/prospective/*_v4`, corrected GS, TJS snapshot |
| 2026 전향 채점 | 위 자료 + 최신 `data/raw/statcast_2026.parquet` |
| M2 golden regression 9/9 재현 | 위 자료 + legacy `gs_flags_v1.parquet` 테스트 fixture |
| MLB feature 전체 재구축 | raw Statcast + label/upstream + base artifact; 아래 portability 주의 참조 |
| A-IL 감사 재현 | `data/ail/transactions_2016..2024.json` + corrected episode |
| Kang Phase 1 정확 재현 | 권한이 확인된 저자 `final_df.csv` 별도 제한 공유 |

`.venv/`, `node_modules/`, cache, log, 과거 prediction dump는 전달하지 않는다.
upstream 디렉터리는 수정하지 않는다. 저자 제공 `final_df.csv`는 upstream
Git 객체에 포함되지 않으며 GitHub에 올리지 않는다.

재현성 주의: 현재 동결 scorer와 M1 corrected 감사 코드는 저장소 내부의
지속 경로를 사용한다. 반면 일부 과거 Phase 2.6/초기 Phase 3 스크립트는
당시 PC의 임시 scratchpad 절대경로를 참조하는 역사적 증거 코드다. 이
구간은 raw Statcast만으로 one-command 전체 재구축이 되지 않는다. 해당
실험을 다시 실행해야 할 때에는 먼저 입력 경로를 portable하게 고치고
필요한 base artifact를 manifest에 추가한다.

## 작업 원칙

- 새 feature·모델은 동결 anchor를 먼저 재현한 뒤 paired delta로 평가한다.
- 모든 입력 feature는 결정일 `t`보다 엄격히 이전(`game_date < t`)이어야 한다.
- MLB 동결 계수, q=0 정책, TJS-only 라벨을 조용히 변경하지 않는다.
- KBO에는 MLB 계수·top-50·quota를 그대로 이식하지 않는다.
- 실패·널 결과와 결과 확인 후 수정된 분석을 숨기지 않는다.
- 원자료·실명 위험 명단·저자 데이터는 공개 Git에 커밋하지 않는다.
- 자동 에이전트는 `git commit`, `push`, `pull`과 외부 배포를 수행하지
  않으며, 저장소 반영은 소유자와 조율한다.

작업을 마치면 `plan_progress.md`에 재현 가능한 결정 로그를 남기고, 모델·결과
수치가 바뀌는 경우 `PROJECT_MEMORY.md`와 관련 정본 문서를 함께 갱신한다.
