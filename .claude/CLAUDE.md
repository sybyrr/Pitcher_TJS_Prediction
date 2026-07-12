# Pitcher TJS Prediction (Kang 2025 재현 → KBO 이전)

MLB 투수의 Tommy John Surgery 위험을 경기 데이터로 예측한 Kang et al. 2025
(*J Big Data* 12:87)를 재현·해체하고, 배치 가능한 전향 위험 지표로 재설계한
뒤, 그 방법론을 근거로 KBO 구단에 데이터 접근을 요청하는 연구 프로젝트.

## 세션 시작 읽기 순서

1. 이 파일 — 상태 한눈에
2. `PROJECT_MEMORY.md` — **공유 가드레일·canonical 수치·기각 이력·인용
   금지 목록** (Claude·codex 공용 단일 원본)
3. `plan_progress.md` — 계획(G1–G3, v3) + 세션별 진행 로그
4. `phase2_results.md` — Phase 2/2.5/2.6/3 결과 canonical (블록 1–9)
5. 상세: `results/phase3/*.md` (최신 수치는 P_BLOCK_RESULTS.md 정정 헤더),
   `Kang_2025_TJS_Prediction.md`(논문 전사), `kang_repo_audit.md`,
   `reproduction_and_dataset.md`, `KBO_applicability.md`

## 목표 (canonical: plan_progress.md G1–G3)

1. **G1 재현 [달성]**: 분류 AUC 0.92 / 회귀 R² 0.783 / SHAP 일치. 이
   baseline은 동결, 모든 수정은 변형으로 대조.
2. **G2 정직한 평가 [조건부 backtest 정리]**: 누수·아티팩트 교정 후
   primary ROC 0.701/0.696. untouched test가 없어 확정적 전향 성능은 아니며,
   E0A safety 경계 민감도와 함께 인용한다 (PROJECT_MEMORY.md 3절).
3. **G3 KBO 이전 [진행]**: 현재까지 데이터만으로 계산 가능한 전향 위험
   지표(순위+근거). 이전 대상은 방법론·파이프라인, **계수는 KBO 재적합**.

## 현재 상태 (2026-07-13)

- **Phase 1 [완료]**: Kang 분류·회귀·SHAP 완전 재현 → `results/`.
- **Phase 2/2.5/2.6 [완료]**: 논문 수치 해체(회귀=개인 내 보간, 분류 ~0.11
  이력길이 아티팩트; 정직 회고 ~0.60) → 전향 재설계(월별 결정일, TJS-only,
  미래 시즌 test)에서 workload LR ROC 0.64 → `phase2_results.md`.
- **Phase 3 [주 개선 캠페인 종료, 최종 검증 블록 대기]**: R(역할 채택·회고 정정) → B/B'(트래킹 증분 널,
  tree 널, 콘텐츠-only ~0.5, 경보의 60%가 사용량 정상 시점) → P(진단 →
  M_sa 채택 → hazard supermodel canonical → 2024 확장) → **codex 외부
  감사 수용(2026-07-13)**: 2025 라벨 미완결로 2022-25 수치 철회, 정정
  canonical = **H90 0.701 / H150 0.696 (조건부 backtest, 사건 75/80)**.
  상세·정정 전문: `results/phase3/P_BLOCK_RESULTS.md`.
- **승인됨(2026-07-13, 사양 v2 = plan_progress.md "2026-07-13 (계속 3)"
  절 — codex 2차 감사 반영·fable 수치 검증 완료)**: **A0(1-3) → A1 불펜
  → A-IL(팔꿈치-IL 이력 feature, MLB) → 동결** 순서로 기계적 실행.
  라벨은 TJS-only 불변(IL은 feature만). primary 0.701/0.696은 safety
  경계 0.660/0.665와 항상 병기; KBO 기대치는 novel 대역 ~0.56–0.68.
- **대기**: 사용자 "시작" 지시; 제안서(사용자 직접 작성).

## Key files

```
PROJECT_MEMORY.md             # 공유 컨텍스트 단일 원본 (가드레일·수치·기각 이력)
AGENTS.md                     # codex 진입점 (이 파일과 함께 갱신)
plan_progress.md              # 계획 + 진행 로그 (블록/세션 종료 시 갱신)
phase2_results.md             # 결과 canonical (블록 1-9, 정정 이력 포함)
results/phase3/               # Phase 3 결과 md + CSV + scripts/ (재현 코드)
Kang_2025_TJS_Prediction.md   # 원 논문 layout-aware 전사본 (수치 기준점)
kang_repo_audit.md            # upstream 코드 감사 + 환경 기록 (함정 목록)
reproduction_and_dataset.md   # 재현 검토 + 데이터셋 정의 (라벨 신뢰: E0A)
KBO_applicability.md          # KBO 이전 전략 검토
src/download_statcast.py      # Statcast 다운로드 (2016-2025) → data/raw/
src/extract.py                # final_df 재구축 (upstream 1:1 + 마커)
src/run_phase2.py             # Phase 2 변형 러너 (VARIANTS 레지스트리)
scripts/verify_env.py         # 환경 검증 (모델 forward + fetch, 학습 X)
TJS_Prediction/               # upstream 클론 — 수정 금지, gitignore
data/                         # raw parquet + prospective v2-v4 — gitignore
.venv/                        # Python 3.11.11 (uv), torch는 cu128 인덱스
```

## 실행

```powershell
.venv\Scripts\python.exe scripts\verify_env.py   # 9/9 통과 기대
# 환경 재구축은 requirements.txt 헤더의 uv 커맨드 참조
```

## 프로젝트 규칙 (상세·근거: PROJECT_MEMORY.md 1–2절)

- `.md` 외 코드·데이터·설정의 생성·수정·다운로드 및 승인된 계획/실험
  실행은 **사용자 명시 지시 후에만** 시작. 읽기 전용 점검과 파일을 남기지
  않는 진단 재계산만 가능.
- `TJS_Prediction/` 수정 금지; 저자 final_df.csv는 GitHub 업로드 절대 금지.
- 라벨 TJS-only 고정 (팀원 트랙과 충돌 방지). arm-IL은 auxiliary feature
  한정 승인, 별도 `시작` 지시 전 실행 금지.
- 외부 레퍼런스는 setup 대조·검증 후 채택, 수치 이식 금지.
- 동결 프로토콜(코호트·fold·평가·anchor 게이트)은 재검토 없이 변경 금지.
- 블록/세션 종료마다 문서 갱신; "md 업데이트" 요청 시 **CLAUDE.md +
  AGENTS.md + PROJECT_MEMORY.md 동시 동기화**.

## Claude 전용 규칙

- 서브에이전트 위임: 가벼운 읽기/조사는 opus, 높은 수준의 추론·설계·종합은
  fable(메인). opus 산출물은 fable이 재검토 후 채택.
- 코드 수정 시 karpathy-guidelines 스킬 로드.
- 자동 memory(`~/.claude/projects/...`)는 PROJECT_MEMORY.md와 내용 동기
  유지 — 둘 중 하나만 갱신하지 말 것.

## 커뮤니케이션·코딩·git·문서 (상세: PROJECT_MEMORY.md 1·7절)

- 한국어 기본. 옵션 매트릭스(A/B/C) + 명시적 권장을 앞에. 가정 선명시.
  대형·비가역 변경은 의도 설명 후 확인 (승인은 맥락 간 이월되지 않음).
  결과는 정직 보고 (실패한 테스트·스킵한 단계 포함).
- 코딩: karpathy 원칙 (가정 표면화 → 최소 → 외과적 → 성공 기준 검증);
  주변 코드의 관용구·주석 밀도 따르기; Python 컨벤션은 PROJECT_MEMORY.md.
- git: 사용자 직접 수행. 에이전트는 명시 요청 시에만 + .gitignore 안전
  유지. Co-Authored-By 금지. 파괴적 플래그는 개별 승인 없이 금지.
- 문서: 결정 메모 스타일, 이력 보존(정정은 덧붙임), 이모지 금지, 진입점은
  이 파일 하나 — 파일별 역할 중복 금지, 공유 수치 교차 일치 확인.
- 환경: Windows 11 / PowerShell, uv(`~/.local/bin/uv.exe`), `.venv/`.
