# PAINS TJS 연구 대시보드 D0

동결된 MLB TJS 위험 모델의 **과거 재현용** q0 순위와 기여 요인을 읽는
제한형 연구 대시보드다. 모델 학습·재보정·데이터 다운로드 기능은 없으며,
현재 선수의 실시간 위험 명단을 제공하지 않는다.

## 입력 경계

- 화면 입력: `public/data/demo_test_top20.csv`
- 원본: `results/phase3/demo_test_top20.csv`의 byte-identical 복사본
- 모델 상태: `results/phase3/frozen_model_state.json`
- 무결성·범위: `public/data/manifest.json`
- 공유 카드: `public/og-research.png` (비문자 추상 연구 그래픽)
- retrospective outcome은 설명용 사후 정보이며 점수 입력이 아니다.

## 로컬 실행

Node.js 22.13 이상이 필요하다.

```powershell
npm ci
npm run dev
```

production 검증:

```powershell
npm run lint
npm test
```

`npm test`는 production build, 서버 렌더, 연구 한계 문구, 동결 hash,
데모 CSV의 byte identity를 확인한다.

`postcss`는 `8.5.14`로 override해 production dependency audit를 0건으로
고정했다. 개발 도구 audit는 별도이며 자동 `audit fix --force`는 금지한다.

## 배포 원칙

`.openai/hosting.json`을 사용하는 Sites 배포 대상으로 구성되어 있다.
배포는 반드시 검증된 source commit과 같은 archive에서 이뤄져야 하며,
초기 접근 정책은 owner-only로 유지한다. 이 저장소의 commit/push는 프로젝트
규칙상 사용자가 직접 수행한다.
