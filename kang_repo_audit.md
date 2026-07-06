# Kang repo 코드 감사 + 환경 재현 기록

작성 기준일: 2026-07-06. 근거: `TJS_Prediction/` 클론본 전 파일 정독(서브에이전트 5개 병렬 스캔) + 로컬 실행 검증.
역할: `reproduction_and_dataset.md`의 "재현 함정" 목록을 실검증하고, 그 문서에 없던 추가 결함과 확정된 환경 스택을 기록한다.

---

## A. 확정된 환경 (검증 스크립트 9/9 통과)

`scripts/verify_env.py` 실행 결과 (2026-07-06):

| 항목 | 논문/README | 우리 환경 | 비고 |
|---|---|---|---|
| Python | 3.11.11 | **3.11.11** | 정확히 일치 (uv 관리) |
| PyTorch | 2.5.1 (CUDA 12.5) | **2.11.0+cu128** | 의도적 상향 — RTX 5060 Ti(sm_120)는 cu128+ 필수 |
| GPU | RTX 4090 / L4 22.5GB | RTX 5060 Ti 16GB | 논문 peak 6.77GB < 16GB, 충분 |
| pandas | 미명시 | **2.3.3 (<3 핀)** | pandas 3.x는 CoW 기본화로 upstream inplace 패턴이 조용히 no-op |
| timm / torchvision | 미명시 | 1.0.27 / 0.26.0 | ViT는 HF hub에서 `augreg2_in21k_ft_in1k` 태그 다운로드 — 논문 당시 가중치와 동일 보장 없음 (추론) |
| pybaseball | 미명시 | 2.2.7 | 2023-05-01 단일일 fetch 검증: 2,317 pitches, Kang 24컬럼 전부 존재 |

검증 내용: 모델 6종(ViT/ResNet50/Transformer/CNN+LSTM/LSTM/회귀 1D-CNN)을 클론본 코드 그대로 import → 논문 하이퍼파라미터로 인스턴스화(pretrained 가중치 다운로드 포함) → GPU dummy forward + weighted-BCE loss 계산. 학습은 하지 않음.

**데이터 접근 현황:**
- `final_df.csv`(전처리 완료본): Google Drive 링크 **HTTP 401 — 비공개**. gdown·직접 요청 모두 거부 (2026-07-06 확인). → raw 재추출이 유일한 경로. (브라우저 "액세스 요청" 또는 저자 메일은 남은 옵션)
- `list of TJ.csv`: repo에 실데이터 있음. 2,436행, 수술연도 1974–2024 (2024년 26건 — 갱신 시점상 불완전 추정). 최신본은 Roegele 공개 시트에서 재다운로드 권장.
- `pitcher_hand.csv`: 실데이터 있음. 1,952행 (RHP 1,488 / LHP 464).

## B. `reproduction_and_dataset.md` 함정 목록 실검증 결과

| # | 문서의 주장 | 검증 결과 |
|---|---|---|
| 1 | final_df.csv가 링크 파일 | **확인** + 링크 자체가 401 비공개 (문서 우려보다 악화) |
| 2 | date_ranges 콤마 누락 → 실행 에러 | **확인, 단 정정**: SyntaxError가 아니라 런타임 TypeError(`tuple object is not callable`). 게다가 그 이전에 L3 `!pip install`이 SyntaxError라 파일 자체가 plain python으로 실행 불가 |
| 3 | 경로 전부 플레이스홀더 | **확인**: 추출 3곳, 분류 2곳, 회귀 5곳 (`...`/`....` 불일치도 있음) |
| 4 | classification_train.py에 `%cd` 매직 | **확인** (L8). regression_train.py L11에도 동일 |
| 5 | sequence 길이 = 224 (주석 128은 오류) | **확인**: bin 필터 (>99 & <1220) → 100..1215, 5일 간격 = 224 timesteps. `== 620` 체크는 결과가 버려지는 no-op |

## C. 문서에 없던 추가 결함 (이번 스캔 신규 발견)

성능에 영향을 줄 수 있는 것 (재현 시 "충실 재현 vs 교정 재현" 분기 대상):

1. **Early stopping이 best weights를 복원하지 못함** — 분류 모델 5종 전부. `best_model_wts = self.state_dict()`가 deepcopy 없이 참조 저장이라, 최종 로드 시 마지막 epoch 가중치가 로드됨. 논문 보고 성능은 이 버그가 있는 채로 측정된 것 (사실: 코드 확인 / 성능 영향 크기: 미지). ViT.py L142, ResNet.py L93, Transformer L139, CNN_LSTM L123, LSTM L96. 회귀는 .pth 파일 저장·재로드 방식이라 무관.
2. **ViT만 train DataLoader `shuffle=False`** (classification_train.py L143) — 다른 4개 모델은 shuffle=True. 의도로 보기 어려움 (추론).
3. **회귀의 player-level leakage 의심** — 부상 투수 전 시점을 행 단위로 합쳐 60/20/20 split (prepforreg.py). 같은 투수의 인접 시점이 train/test에 갈라져 들어감 → R² 0.79는 낙관적일 수 있음 (구조는 사실, 영향은 추론 — 재현 후 정량화 대상).
4. **ResNet만 pos_weight를 tensor로 감싸지 않음** (ResNet.py L24) — raw int를 넘기면 TypeError. 학습 스크립트가 `torch.tensor(5)`(int64)를 넘겨서 동작. 로컬 검증으로 확정.
5. **ResNet 조기 종료 시 `load_state_dict(None)` 크래시 경로** (ResNet.py L99) — 개선 전 patience 도달 시. 다른 4개 파일은 None 가드 있음.
6. **main()은 CNN_LSTM·LSTM만 실행** — ViT/ResNet/Transformer 호출은 주석 처리 (classification_train.py L429-431). 논문 주력 모델(ViT)을 돌리려면 주석 해제 필요.
7. **`list of TJ.csv` 품질 이슈**: 날짜 포맷 2종 혼재(`2022-8-1` vs `6/3/2004`), 비투수(C/OF) 포함, 다중 수술자 중복 행, 선택 컬럼 결측 다수. 파서·필터에서 처리 필요.

실행 가능성 관련 (파이프라인 재작성 근거):

8. `train_for_reg.py`는 `from 1d_cnn import CNN`(숫자 시작 모듈명) SyntaxError — 죽은 레거시. 진입점은 regression_train.py.
9. pandas 2.2+/3.0 취약 패턴 다수: chained `inplace=True`(추출 L303 — CoW에서 조용히 NaN 잔존), `groupby.apply`의 include_groups deprecation (추출 L258, Prepfortrain L84), 필터 슬라이스 위 할당 (여러 곳). pandas<3 핀은 이것 때문.
10. L1 정규화가 pretrained backbone 전체 파라미터(ViT ~86M)에 매 배치 적용 — 성능·속도상 특이 설계 (사실, 의도 여부는 미지).

## D. 파일별 역할 지도 (재작성 시 참조)

```
Raw_data_extraction/Pybaseball_extract.py  # statcast 추출→pivot→TJ 라벨 join→좌완 보정→보간→diff→final_df.csv. 노트북 export라 그대로는 실행 불가
Classification/Prepfortrain.py             # final_df.csv→±4.7σ 아웃라이어→5일 bin→(N,224,102) X 생성 + 6:2:2 split/MinMax
Classification/classification_train.py     # 10 seed(100..1000) 학습 루프. 진입점 (단 %cd 제거 필요)
Classification/{ViT,ResNet,...}.py         # 모델+train_model 스캐폴드 (import 가능, 검증 완료)
Regression/prepforreg.py                   # target==1만, <550일, 5/10일 bin, (N,1,feat)
Regression/regression_train.py             # 진입점: seed 102..1002, 학습+SHAP(GradientExplainer)
Regression/regression_cnn.py               # 1D-CNN 4conv+4fc. Adam lr=0 + warmup 스케줄러(eta_max가 실제 LR)
Regression/train_for_reg.py                # 죽은 레거시 (import 불가) — 무시
```

## E. 판정

- **환경 재현: 완료.** 모델 코드는 최신 스택에서 무수정 동작 확인 (deprecation 경고 수준).
- **데이터 파이프라인: 재작성 필수.** 추출 스크립트는 노트북 유물로 실행 불가 + 전처리본 링크 비공개. `reproduction_and_dataset.md`의 raw 재추출 권장이 유일 경로로 확정됨.
- **재현 실험 설계 시 분기**: (a) 버그 포함 충실 재현(논문 수치와 비교 목적) → (b) 교정 재현(deepcopy·shuffle·leakage 수정)의 성능 차이 자체가 기여가 될 수 있음.
