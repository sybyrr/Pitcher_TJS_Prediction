# Pitcher TJS Prediction (Kang 2025 재현 → KBO 적용)

MLB 투수의 Tommy John Surgery 위험을 경기 데이터만으로 예측한 Kang et al. 2025
(*J Big Data* 12:87)를 재현하고, 최신 데이터로 확장한 뒤, 그 결과를 근거로
KBO 구단에 트래킹 데이터 접근을 요청하는 것이 최종 목표인 연구 프로젝트.

Read in this order on session start for fast context:
1. This file — current status at a glance
2. `plan_progress.md` — 승인된 Phase 0–3 계획 + 세션별 진행 로그
3. `Kang_2025_TJS_Prediction.md` — 원 논문 전사본 (문제 정의 + 수치 기준)
4. `kang_repo_audit.md` — 클론 코드 감사 + 확정 환경 기록
5. `reproduction_and_dataset.md`, `KBO_applicability.md` — 데이터 정의, KBO 전략

## Goal

1. **재현**: Kang과 동일 정의(2016–2023)로 분류 F1 ~0.73 / ROC-AUC ~0.93,
   회귀 R² ~0.79 수준 재현. 이것이 baseline 신뢰의 성공 기준.
2. **확장**: +2024 데이터 검증, 교정 재현(발견된 버그 수정판)과의 비교.
3. **KBO 제안**: feature ablation으로 "KBO 비공개 feature의 기여도"를 정량화한
   구단 설득 자료. 2025–2026 데이터는 v1 제외 (plate_x/z 정의 변경 + 라벨 미성숙).

## Current status (2026-07-06)

**Phase 1 충실 재현 완료 (SHAP만 잔여).** 분류 5모델×10 seeds: ViT 0.71/0.92
(논문 0.73/0.93), ResNet·Transformer·CNN+LSTM은 논문과 일치, 순위 동일.
회귀 R² 0.783±0.019 (논문 0.79), 100-Day RMSE 94.6 (논문 95.7). 성공 기준
(F1 ±0.05) 충족. 결과: `results/*.csv`. 잔여: 회귀 SHAP (체크포인트
`data/checkpoints/`에서 재학습 없이 실행, `src/train_regression.py` SHAP 플래그).
다음: SHAP → Phase 2 (버그 교정, player-level split, 시간적 외부검증).
Phase 0 상세(재추출본 검증 r=0.978 등)는 `plan_progress.md`.

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

- Do NOT add a `Co-Authored-By: Claude ...` line to commit messages.
- Commit or push only when asked. If on the default branch, branch first.
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
