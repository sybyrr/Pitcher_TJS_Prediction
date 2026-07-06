# Kang 2025 재현 검토 + 데이터셋 정의 (Claude Code 준비 문서)

작성 기준일: 2026-07-01. 근거: Kang 논문 + GitHub `dxlabskku/TJS_Prediction` 코드 정독 + pybaseball/Baseball Savant 문서 확인.
목적: RTX 5060 Ti 16GB로 재현 가능한지, 그리고 지금 시점에 어떤 데이터를 불러올지 정의. 실제 구현은 Claude Code에서.

---

## A. GPU 재현성 검토 — 결론: **가능. 16GB로 충분.**

### 근거 (코드에서 직접 확인한 수치)

가장 무거운 모델이 요구량 상한을 정한다. 코드상 최댓값은 이렇다.

| 모델 | batch | 입력 크기 | 논문 보고 peak GPU | 판정 (16GB 대비) |
|---|---|---|---|---|
| ViT (vit_base_patch16_224, pretrained) | 16 | 224×224×3 | **5.18 GB** | 여유 |
| ResNet-50 (1ch→conv1 교체) | 16 | 224×224×1 | 1.29 GB | 여유 |
| Transformer-Encoder (512d, 16head, 12layer) | 64 | 224×102 | 6.77 GB | 여유 |
| CNN+LSTM (512d, 4layer) | 64 | 224×102 | 2.79 GB | 여유 |
| LSTM (512d, 4layer) | 128 | 224×102 | 5.74 GB | 여유 |

논문 보고 peak가 최대 6.77GB(Transformer). 16GB의 절반 이하다. 원 저자 하드웨어가 22.5GB(RTX 4090 / L4)였지만 실제 사용량은 그보다 훨씬 작다.

### 5060 Ti 특유의 확인 사항 (Blackwell, sm_120)
- 5060 Ti는 최신 아키텍처(Blackwell)라 **CUDA 12.8+ / 그에 맞는 PyTorch 빌드**가 필요하다. 논문 스택(CUDA 12.5, torch 2.5.1)을 그대로 쓰면 sm_120 미지원으로 안 돌 수 있다. → Claude Code 단계에서 최신 cu128 계열 torch로 설치할 것. (모델 코드 자체는 표준 nn 모듈이라 버전만 맞추면 그대로 동작.)
- 데이터셋이 620개로 매우 작아 연산량 자체가 가볍다. 병목은 GPU 메모리가 아니라 (1) 초기 Statcast 다운로드 시간, (2) pretrained 가중치 다운로드(ImageNet ViT/ResNet).

### 시간 비용 (참고)
- 논문 기준 ViT 학습 217s/seed, 10 seed → 모델 하나 ~36분. 5060 Ti는 4090보다 느리므로 여유 있게 보면 ViT 10-seed ~1~2시간. 5개 모델 전부 10-seed라도 하루 안쪽.
- 실제 최대 병목은 학습이 아니라 **2016–2023 전체 Statcast pitch-level 다운로드**(수백만 행, Baseball Savant는 쿼리를 1일 단위로 쪼갬). 캐시 켜고 한 번 받아두는 게 핵심.

### 재현 시 주의 (코드에서 발견한 함정 — Claude Code에 넘길 것)
1. **`final_df.csv`가 repo에 실데이터로 없음.** Google Drive 링크 파일이다. 즉 전처리 완료본을 그대로 받거나, `Pybaseball_extract.py`로 raw부터 다시 만들어야 한다.
2. **`Pybaseball_extract.py`의 `date_ranges` 리스트에 문법 오류(콤마 누락)** — 2020/2021 튜플 사이 콤마가 빠져 있어 그대로 실행하면 에러. 수정 필요.
3. **경로가 전부 `'.../final_df.csv'` 플레이스홀더** — 실제 경로로 교체 필요.
4. **`classification_train.py`에 Jupyter 매직(`%cd`)** 포함 — .py 실행 시 제거 필요.
5. **전처리 주석 `# each sequence contains 128 datapoints` 와 `== 620` 체크가 서로/필터와 불일치.** 필터 `> 99 & < 1220`을 5일 격자로 세면 sequence 길이 = **224** (100,105,…,1215). 실제 `X.shape`를 print로 확인 후 진행.
6. **ViT/ResNet 이미지화:** feature 102 → `pad_to_224`가 `-5`로 우측 패딩. time축은 이미 224. `x.view(b,1,224,224)` 후 3채널 복제. (224 설계는 ViT 입력에 맞춘 것.)
7. **결정론:** `set_seed` + `cudnn.deterministic=True` 설정돼 있으나, seed 리스트가 `range(100,1001,100)` = 10개 seed.

### 판정
16GB는 **넉넉하다.** 진짜 리스크는 메모리가 아니라 (a) 5060 Ti용 torch 버전 맞추기, (b) 데이터 파이프라인 재구성(위 함정 1~5). Claude Code에서 raw 재추출 경로를 택하는 걸 권장 — final_df.csv 링크는 언제 죽을지 모르고, 최신 데이터로 확장하려면 어차피 추출 코드가 필요하다.

---

## B. 데이터셋 정의 — 근거 있는 현재 시점 권장

원칙: **재현성**(Kang과 같은 정의로 먼저 맞춰보기) + **시의성**(2026년 현재 최신·유효한 소스). 직접 만들 필요 없이 "무엇을 어떤 패키지로 불러올지"만 정의한다.

### B-1. 소스 3개 (전부 public, 무료)

| 소스 | 내용 | 불러오는 법 | 현재 상태 |
|---|---|---|---|
| **Statcast (Baseball Savant)** | pitch-level 궤적 metric | `pybaseball.statcast(start_dt, end_dt)` | 2008~ 유효, 2015+ 최신 metric 포함. Blackwell 무관, 그냥 스크래핑 |
| **Tommy John Surgery list** | 수술 라벨 (선수, 날짜, level) | Roegele 공개 Google Sheet (repo의 `list of TJ.csv`가 스냅샷) | 여전히 유지·갱신 중. 최신본으로 재다운로드 권장 |
| **투수 손잡이** | LHP/RHP (좌완 보정용) | repo `pitcher_hand.csv` 또는 pybaseball player lookup | 정적 |

### B-2. 시즌 범위 — 권장: **재현용 2016–2023 + 확장용 2016–2024**, **2025–2026은 제외**

근거 (시의성 vs 재현성):
- **재현 1차: 2016–2023 정확히.** Kang과 동일 정의로 먼저 F1 0.73 / ROC-AUC 0.93이 재현되는지 확인. 이게 baseline 신뢰의 출발점.
- **확장 2차: 2024 추가(2016–2024).** 표본을 늘려 검증. 2024는 metric 정의가 2016–2023과 동일해 안전하게 붙는다.
- **2025–2026은 v1에서 제외 — 이게 이번 조사의 핵심 시의성 발견:**
  - **plate_x / plate_z 정의 변경.** Baseball Savant 문서 확인: **2025년까지 front-of-plate 기준, 2026년부터 ABS 시스템에 맞춰 middle-of-plate로 변경.** Kang의 102 feature에 `plate_x_*`·`plate_z_*`가 구종별로 들어가므로, 2026 데이터를 섞으면 이 두 계열이 이전 시즌과 **정의가 달라 시계열이 깨진다.** 반드시 분리하거나, plate_x/z를 제외하거나, 별도 정규화해야 함.
  - **2025–2026 TJS 라벨 미성숙.** Kang 설계는 "수술 전 궤적"을 본다. 2025~2026 수술자는 아직 라벨/추적이 불완전하고, 최근 시즌일수록 "아직 부상 안 한 것처럼 보이는 미래 부상자"가 non-injury에 섞일 우려(right-censoring)가 크다.
  - 결론: **2026 데이터는 v1 재현·검증에서 빼고**, 나중에 plate_x/z 처리 방침을 정한 뒤 별도로 붙인다. (이건 사용자 CLAUDE.md 메모의 "2026 ABS plate_x 변경 주의"가 실제로 맞았음을 확인한 것.)

### B-3. 컬럼 — Kang과 동일하게 시작

`statcast()`가 80+ 컬럼을 주지만, 아래 24개만 선택(코드 `select_columns`와 동일):

```
player_name, game_type, home_team, pitch_type, game_date, pitcher, game_year,
release_speed, release_pos_x, release_pos_z, pfx_x, pfx_z, plate_x, plate_z,
vx0, vy0, vz0, ax, ay, az, effective_speed, release_spin_rate,
release_extension, spin_axis
```
- 필터: `game_type == 'R'` (정규시즌).
- pivot: index = (player_name, home_team, game_date, pitcher, game_year), columns = pitch_type, 17 metric을 mean 집계.
- 구종 6개(CH/CU/FC/FF/SI/SL)만 생존(>90% 결측 or 단일값 컬럼 제거) → 17×6 = **102 feature**.
- **주의:** `arm_angle`은 Kang이 안 씀(2020+만 존재). Mastroianni add-on을 나중에 붙일 때만 관련되고, Kang 재현에는 불필요.

### B-4. 라벨/코호트 정의 (코드 기준, 그대로 유지)

- **target = 1** if `TJ Surgery Year` not null (해당 투수가 TJS 받음), else 0.
- TJS 그룹: 수술 전 마지막 2시즌 사용, 최소 2경기 등판.
- non-injury 그룹: 최근 4시즌 중 3+ 시즌 참여(코드에선 4연속 시즌 확보 로직).
- **최종 표본: 620 (injured 101 / non-injured 519).** 확장 시 이 숫자가 늘어남.
- day 0 = 마지막 경기(부상자는 수술 직전 마지막 경기). 5일 bin. 분류 window = 부상 100~1215일 전(직전 100일 제외).

### B-5. Claude Code에 넘길 "불러오기" 요약

```python
# 1) Statcast (재현: 2016–2023, 확장: +2024). 2025–2026 제외.
from pybaseball import statcast
from pybaseball import cache; cache.enable()   # 대용량 → 캐시 필수
# 연도별 (YYYY-03-01 ~ YYYY-11-30) 정규시즌 범위로 나눠 받기

# 2) TJS 라벨: Roegele 공개 시트 최신본 CSV
#    (repo의 'list of TJ.csv'로 먼저 재현 → 이후 최신본으로 교체)

# 3) 손잡이: repo 'pitcher_hand.csv' 또는 playerid_lookup

# 주의: date_ranges 콤마 오타 수정, %cd 제거, 경로 교체,
#       plate_x/plate_z는 2016–2024 안에서만 (2026 정의변경 회피).
```

---

## C. 한 줄 요약

- **GPU:** RTX 5060 Ti 16GB로 재현 **가능**(peak ~6.8GB). 진짜 리스크는 메모리가 아니라 Blackwell용 torch 버전 + 데이터 파이프라인 재구성.
- **데이터:** 재현은 **2016–2023**(Kang 동일), 확장은 **+2024**. **2025–2026 제외** — 2026 plate_x/z 정의가 ABS로 바뀌어 시계열이 깨지고 최신 시즌 라벨이 미성숙하기 때문.
