# Pitcher TJS Prediction (Kang 2025 재현 → KBO 적용)

MLB 투수의 Tommy John Surgery 위험을 경기 데이터만으로 예측한 Kang et al. 2025
(*J Big Data* 12:87)를 재현하고, 최신 데이터로 확장한 뒤, 그 결과를 근거로
KBO 구단에 트래킹 데이터 접근을 요청하는 것이 최종 목표인 연구 프로젝트.

Read in this order on session start for fast context:
1. This file — current status at a glance
2. `plan_progress.md` — 계획(v2, G1–G3) + 세션별 진행 로그
3. `phase2_results.md` — Phase 2/2.5 결과 canonical (변형표, 전향 결과)
4. `phase2_findings.md` — 검증 발견 36건 / `phase2_worklog.md` — 실행 아카이브
5. `Kang_2025_TJS_Prediction.md`(논문), `kang_repo_audit.md`(코드 감사·환경),
   `reproduction_and_dataset.md`, `KBO_applicability.md`

## Goal (v2, 2026-07-06 개정 — canonical은 plan_progress.md의 G1–G3)

1. **G1 재현** [달성]: Kang 동일 정의로 분류 F1 0.71/AUC 0.92, 회귀 R² 0.783
   재현 완료. 이 baseline은 동결, 모든 수정은 변형으로 대조.
2. **G2 정직한 평가**: 누수·아티팩트 교정 후 배치 상황(새 투수·미래 시즌)의
   성능 확정. 수치 하락은 실패가 아니라 산출물.
3. **G3 KBO 이전**: 최종 산출물은 **현재까지의 데이터만으로 계산 가능한
   전향적 부상 위험 지표**(위험도 순위 + 역학 근거). 전향적 재설계(Phase 2.5)가
   전제 조건이며, feature-tier ablation은 교정된 파이프라인 위에서 수행.
   2025–2026 데이터는 제외 (plate_x/z 정의 변경 + 라벨 미성숙).

## Current status (2026-07-07)

**Phase 1 완전 종료 — 분류·회귀·SHAP 전부 재현.** 분류: ViT 0.71/0.92(논문
0.73/0.93), 나머지 4모델 논문과 일치, 순위 동일. 회귀: R² 0.783±0.019(논문
0.79), RMSE 94.6(논문 95.7). SHAP: distinct 20개 동일, 평균 |SHAP| top-3 동일
집합(release_speed_FF·spin_axis_FF·release_extension_SL). 결과: `results/`.
**Phase 2 + 2.5 완료 (2026-07-07 야간 자율 실행).** 결과 canonical:
`phase2_results.md`. 3줄: ① 회귀 신투수 R² 음수(논문 0.79는 within-pitcher
보간) ② 정직한 회고 분류 수치 = **AUC 0.816** (fill-artifact 제거 × 미래
시즌; 논문 0.93 중 ~0.11은 이력길이 아티팩트) ③ 전향(rolling) 예측은 현
설계에서 무작위 수준. **Phase 3 진입 전 사용자 결정 대기**: 전향 설계 반복
vs 회고 프레이밍 전환 (`phase2_results.md` 마지막 절). 상세: `plan_progress.md`.

## Key files

```
plan_progress.md              # Phase 0–3 계획 + 진행 로그 (세션 종료 시 갱신)
Kang_2025_TJS_Prediction.md   # 원 논문 layout-aware 전사본 (수치 기준점)
kang_repo_audit.md            # 코드 감사 + 환경 기록 (함정 목록의 canonical 소스)
reproduction_and_dataset.md   # 재현 검토 + 데이터셋 정의 (2016–2023 / +2024 / 2025–26 제외)
KBO_applicability.md          # KBO 이전 전략 검토
src/download_statcast.py      # Statcast 시즌별 다운로드 → data/raw/*.parquet
src/extract.py                # final_df 재구축 (upstream 1:1 + [GAP-FILL]/[DEVIATION] 마커)
src/prep_classification.py    # final_df → X(N,224,103)/y 텐서 (Prepfortrain 포트)
src/compare_final_df.py       # 재추출본 vs 저자본 정량 비교
src/run_phase2.py             # Phase 2 분류 변형 러너 (VARIANTS 레지스트리 canonical;
                              #   전체 Phase 2/2.5 코드 목록은 phase2_results.md 산출물 인덱스)
results/                      # Phase 1 재현 + phase2/ 변형 결과 CSV
scripts/verify_env.py         # 환경 검증 (모델 forward + statcast fetch, 학습 X)
TJS_Prediction/               # upstream 클론 (수정 금지, gitignore) + Raw_data/final_df.csv(저자본)
data/                         # raw parquet, 재추출 final_df (gitignore됨)
requirements.txt              # 직접 의존성 (+ requirements.lock.txt 전체 핀)
.venv/                        # Python 3.11.11 (uv 관리, gitignore됨)
```

## How to run

```powershell
.venv\Scripts\python.exe scripts\verify_env.py   # 환경 검증 (9/9 통과 기대)
# 환경 재구축이 필요하면 requirements.txt 헤더의 uv 커맨드 참조
```
env: `.venv/` (uv 기반, Python 3.11.11). torch는 반드시 cu128 인덱스에서 설치.

## Project rules

- **학습(training) 실행은 사용자 명시 요청 후에만 시작한다.**
- `TJS_Prediction/`(upstream 클론)은 수정하지 않는다. 재현 코드는 별도
  디렉토리(`src/` 예정)에 작성하고, upstream과의 차이를 문서화한다.
- 대량 Statcast 다운로드는 `pybaseball.cache.enable()` 켜고 진행.
- 코드 수정 시 karpathy-guidelines 스킬을 로드한다. 서브에이전트 위임 시
  가벼운 읽기/조사는 opus, 높은 수준의 추론·설계는 fable(메인)로.

---

<!-- ===== WORKING AGREEMENT — task-agnostic, keep across projects ===== -->

## Communication

- Respond in Korean by default.
- Prefer an option matrix (A / B / C) with an explicit recommendation over an
  open-ended explanation. Lead with the recommended choice.
- State assumptions up front. Explain intent and confirm before large or
  hard-to-reverse changes; approval in one context does not extend to the next.
- Report outcomes faithfully: if tests fail, say so with the output; if a step
  was skipped, say so. State verified results plainly without hedging.
- Environment: Windows 11 / PowerShell. Python via uv (`~/.local/bin/uv.exe`),
  project venv at `.venv/`.

## Coding conventions

- Follow the [karpathy guidelines](https://x.com/karpathy/status/2015883857489522876):
  surface assumptions → minimal code → surgical (touch only what's needed) →
  define and verify success criteria before declaring done.
- Prefer a generalizable algorithm over ad-hoc, one-off rules.
- Match the surrounding code's idiom, naming, and comment density.
- Python: `from __future__ import annotations` + type hints; `@dataclass(frozen=True)`
  where it fits; snake_case; one-to-two-line function docstrings.
- OS-independent paths via `Path(__file__).parent` — relative, never hardcoded.

## Git

- The user runs commit/push/pull themselves. Claude touches git state only on
  an explicit per-instance request (keeping .gitignore safe stays Claude's job).
- Do NOT add a `Co-Authored-By: Claude ...` line to commit messages.
- If asked to commit on the default branch, branch first.
- Before any risky git operation (merge, rebase, pull, reset):
  1. Commit current work first as a rollback checkpoint.
  2. Name the specific files that will conflict.
  3. Use branch-based / `--no-ff` workflows. Never run destructive flags
     (`reset --hard`, `pull -X theirs`, force-push) without explicit
     per-instance approval.

## Documentation

- Write decision memos, not reference manuals: state each principle once and
  link to the canonical source (code / commit / config) for the details.
  Ask "manual or memo?" → choose memo.
- Source-based — verify directly. Mark fact vs. inference explicitly.
- Formal register, no emoji. Keep docs compact; preserve version/decision
  history (e.g. v2 → v3 → v4) rather than overwriting it.
- One entry-point file (this one). Keep per-file roles distinct and avoid
  duplicating across docs; cross-check shared numbers stay consistent.
- Update the entry point and progress doc at the end of each stage/session.
