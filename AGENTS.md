# AGENTS.md — codex 진입점 (Pitcher TJS Prediction)

MLB 투수 TJS 위험 예측 연구: Kang 2025 재현·해체 → 전향 위험 지표 재설계 →
KBO 이전. 이 파일은 codex용 진입점이며, Claude Code의 진입점
`.claude/CLAUDE.md`와 **같은 원본을 공유**한다. 두 에이전트의 공통 전제는
전부 `PROJECT_MEMORY.md`에 있다 — **작업 전 반드시 먼저 읽을 것.**

## 읽기 순서

1. `PROJECT_MEMORY.md` — 작업 원칙(사용자 지시 이력), 동결 프로토콜,
   **canonical 수치와 인용 금지 목록**, 기각 이력, 현재 계획, 데이터 지도
2. `plan_progress.md` — 계획(G1–G3) + 세션별 진행 로그 (최신 절이 현재
   상태; 승인 블록의 canonical 실행 사양 = "2026-07-13 (계속 3)" 사양 v2)
3. `phase2_results.md` — 결과 canonical (블록 1–9; 정정은 덧붙임 표기)
4. `results/phase3/FROZEN_MODEL.md` — **동결 모델·인용 수치의 최종 단일
   원본** (2026-07-13 동결: 계수·재보정·경보 정책·전향 확인 프로토콜).
   수치 정정 이력은 P_BLOCK_RESULTS.md, 민감도는 PROJECT_MEMORY.md 3절.
5. 필요 시: `results/phase3/*.md`(블록별 상세), `kang_repo_audit.md`(upstream
   함정 목록), `reproduction_and_dataset.md`(데이터 정의·라벨 신뢰성)

## 절대 규칙 (PROJECT_MEMORY.md 1–2절 요약; 상세는 그쪽이 canonical)

- `.md`는 에이전트 판단으로 갱신 가능하지만, 그 외 코드·데이터·설정의
  생성·수정·다운로드와 승인된 계획/실험 실행은 **사용자 명시 지시 후에만**
  시작한다. 읽기 전용 분석·파일을 남기지 않는 검증 재계산은 가능하다.
- git commit/push/pull은 사용자 직접. `.gitignore` 안전 유지(특히
  `data/`, `TJS_Prediction/` — 저자 final_df.csv 업로드 절대 금지).
- **라벨은 TJS-only 고정** (팀원의 distance-based 트랙과 충돌 방지;
  IL 라벨 확장 금지). arm-IL feature는 A-IL 블록에서 검정 완료 —
  blackout 게이트 실패로 **canonical 금지, triage 재검정 후보로만**
  (AIL_RESULTS.md 정정 헤더 참조).
- `TJS_Prediction/`(upstream 클론) 수정 금지.
- 동결 프로토콜(코호트·fold·clustered bootstrap seed 0·anchor 게이트·
  paired 판정) 변경 금지. 새 변형은 게이트 재현 후 paired delta로만 평가.
- 외부 레퍼런스는 setup 대조 후 borrow/adapt/avoid 명시, 수치 이식 금지.
- 수치 인용 시 PROJECT_MEMORY.md 3절의 인용 금지 목록 준수
  (예: Kang 0.93, "무작위 2–3배", 철회된 0.689/0.693).

## 작업 후 의무

- 블록/세션 종료 시 `plan_progress.md` 로그 갱신, 결과는 결정 메모
  스타일로 `results/`에 (스크립트 + CSV + md, 재현 가능하게).
- 사용자가 "md 파일 업데이트"를 요청하면 **PROJECT_MEMORY.md +
  .claude/CLAUDE.md + 이 파일을 함께 동기화**한다 (수치·상태·대기 결정이
  세 파일에서 어긋나지 않게).
- 문서 정정은 원문 삭제가 아니라 정정 표기 덧붙임 (이력 보존).

## 환경

- Windows 11 / PowerShell. Python: uv 관리 `.venv/`(3.11.11) —
  `.venv\Scripts\python.exe`로 실행. torch는 cu128 인덱스.
- 환경 검증: `.venv\Scripts\python.exe scripts\verify_env.py` (9/9 기대).
- 한국어로 응답. 옵션 매트릭스 + 명시적 권장. 결과는 정직 보고
  (실패·스킵 포함). 스타일 상세: PROJECT_MEMORY.md 7절.
