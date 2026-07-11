# 제주항공 B737 예지정비 AI PoC

미래내일 일경험 프로젝트형 — 제주항공 협업
수행 기간: 2026년 6월 ~ 진행 중

---

## 프로젝트 개요

항공기 엔진의 잔여 수명(RUL, Remaining Useful Life)을 AI로 예측해
정비 시점을 사전에 최적화하는 예지정비 시스템 구현

**현재 문제**
- 고장 후 정비 → 갑작스러운 결함 위험
- 주기 정비 → 멀쩡한 부품도 교체하는 과잉 정비 비용

**목표**
- 센서 데이터 기반 RUL 예측
- 특히 제주항공 LCC(단거리 다회 운항) 조건에 특화된 모델

---

## 버전 현황

이 저장소는 하나의 완성된 모델이 아니라, **베이스라인 → 버그 수정 → 구조 전면 개편** 순으로 발전해온 과정을 그대로 남겨두는 것을 목표로 한다. 각 버전의 코드·문서는 지우지 않고 버전별로 보존한다.

| 버전 | 날짜 | 한 줄 요약 | 상태 |
|---|---|---|---|
| **v1.0** | 06-26 ~ 06-29 | 앙상블(GBM+RF+MLP) + CNN-LSTM 베이스라인 | 🟢 완료 |
| **v1.1** | 06-30 | 팀 피드백 반영 + 데이터 누수 등 버그 5건 수정 | 🟡 부분 실행 |
| **v2 (v12)** | 07-01 | 멀티헤드 CNN 기반 8단계 파이프라인(불확실성·고장분류·conformal) | 🟡 미실행 (코드 작성 완료, 기록용) |

버전별 상세 변경 내역은 [`CHANGELOG.md`](./CHANGELOG.md)에 정리되어 있다. **가장 최근 실행 결과와 다음에 할 일은 항상 `results/` 안의 버전별 문서를 최신 기준으로 볼 것.**

---

## 데이터셋

NASA N-CMAPSS (New Commercial Modular Aero-Propulsion System Simulation)
- (17) Turbofan Engine Degradation Simulation-2 선택하여 다운로드.

| 파일 | 엔진 수 | 비행 클래스 | 다운로드 |
|-----|--------|-----------|--------|
| DS01-005.h5 | 5개 | 단/중/장거리 | [nasa.gov](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| DS02-006.h5 | 6개 | 장거리 | [nasa.gov](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| DS03-012.h5 | 12개 | 단/중/장거리 | [nasa.gov](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |


> ⚠️ h5 파일은 용량이 커서 (파일당 약 3GB) 이 저장소에 포함되지 않습니다.
> 위 링크에서 직접 다운로드 후 Google Drive에 업로드해서 사용하세요.

---

## 파일 구조

```
jejuairline/
├── README.md                          이 파일 — 전체 개요, 버전 현황, 실행법
├── CHANGELOG.md                       버전별 변경 이력 (v1.0 → v1.1 → v2)
│
├── notebooks/                         실행 가능한 Colab 원본 노트북
│   ├── 260630-v1_1.ipynb                v1.1 — 피드백 반영·버그 수정 (부분 실행)
│   ├── 260701_8stage_cap50.ipynb        v2 — 8단계 파이프라인, RUL 상한 50 (미실행)
│   └── 260701_8stage_cap125.ipynb       v2 — 8단계 파이프라인, RUL 상한 125 (미실행)
│
├── src/                               v1.0 베이스라인의 재사용 가능한 스크립트
│   ├── final_baseline.py                v1.0 최종 실험 코드 (WINDOW=50, LCC=5배, CNN 3층)
│   ├── save_final.py                    v1.0 실험 결과 + 모델 전체 저장
│   └── load_final.py                    v1.0 다음날 이어하기용 로더
│
└── results/                           버전별 결과·분석 문서
    ├── results_summary.md               v1.0 실험 결과 종합 보고서
    ├── team_comparison.md               v1.0 매뉴얼 대비 비교 + 팀 질문
    ├── v1.1_bugfix_and_results.md       v1.1 버그수정 내역 + 실제 결과 + 핵심 발견
    └── v2_8stage_design_note.md         v2 구조 설계 기록 (미실행, 기록용)
```

**왜 `src/*.py`와 `notebooks/*.ipynb`가 같이 있나?** v1.0은 이미 정리된 스크립트(`src/`)로 존재해서 그대로 뒀다. v1.1부터는 노트북 자체에 단계별 설명·실행 로그·수정 이력이 markdown 셀로 촘촘히 남아 있어서, 이걸 `.py`로 다시 뽑아내기보다는 **Colab 노트북 원본 그대로 보존**하는 쪽을 택했다. GitHub에서 `.ipynb`는 markdown과 실행 결과가 그대로 렌더링된다.

---

## 실행 방법

### v1.0 (완료된 베이스라인 재현)
```python
# 1. Google Colab에서 src/final_baseline.py 전체 복붙 후 실행
FOLDER = '/content/drive/MyDrive/제주항공'  # 본인 경로로 변경

# 2. 결과 저장 → src/save_final.py 실행
# 3. 다음날 이어하기 → src/load_final.py 실행 (모든 모델/데이터 복원)
```

### v1.1 (버그 수정판 — 이어서 실행 필요)
```
notebooks/260630-v1_1.ipynb 를 Colab에서 열고 위→아래로 실행.
DS03-012 중거리(get_or_run_class(FILES[2], 2))·장거리(fc=3)와
마지막 종합 시각화 셀이 아직 안 돌아갔으니 이어서 실행할 것.
```

### v2 / v12 (8단계 파이프라인 — 아직 미실행)
```
notebooks/260701_8stage_cap50.ipynb 와 260701_8stage_cap125.ipynb 를
각각 Colab에서 순서대로 실행: 정의 셀 전체 → prepare() → run_fold(k) 반복 → finalize()
fold·모델 단위로 체크포인트가 저장되므로 세션이 끊겨도 이어서 실행 가능.
CAP=50과 CAP=125는 체크포인트 폴더가 분리되어 있어 결과가 섞이지 않지만,
RMSE 스케일이 다르므로 두 버전 결과를 직접 비교하지 말 것.
```

---

## 핵심 결과 요약

> 세부 표와 그래프는 각 버전 문서를 참고. 여기서는 버전을 넘나드는 결론만 요약한다.

- **장거리 예측이 가장 쉽다.** 엔진 수가 충분한 DS02-006 장거리는 v1.0(9.57) → v1.1(8.60)로 갈수록 CNN-LSTM 성능이 좋아졌다.
- **LCC(단거리) 보정 효과는 v1.0에서 과대평가돼 있었다.** v1.0은 "가중치를 주면 15~20% 개선된다"고 봤지만, v1.1에서 데이터 누수를 제거하고 다시 재보니 개선 효과가 거의 사라졌다(자세한 내용: [`results/v1.1_bugfix_and_results.md`](./results/v1.1_bugfix_and_results.md)). **LCC 보정은 가중치를 주는 방식이 아니라 다른 접근(예: Medium+Long 선학습 후 Short fine-tuning)이 필요할 수 있다.**
- **단거리 결과의 큰 폴드 편차는 모델 문제가 아니라 엔진 수(2~3개) 자체의 한계**라는 점을 v1.1에서 `cross_val()`이 표준편차·엔진 수를 함께 출력하도록 바꿔서 명확히 했다.
- **v2는 "RUL 하나만 잘 맞히기"에서 "얼마나 확신하는지 + 어디가 고장인지까지 같이 보기"로 목표 자체를 넓힌 버전**이다. 아직 실행 전이라 v1과 성능을 직접 비교할 수는 없다.

---

## 모델 구조

### v1.0 / v1.1 — 앙상블 + CNN-LSTM (비행클래스별 분기)
```
입력: 센서 데이터 (WINDOW=50 슬라이딩 윈도우)
         ↓
[비행 클래스 판별]
         ↓
단거리 → LCC 보정 앙상블 (GBM + RF + MLP)      ← v1.1에서 보정 효과 재검증 중
중거리 → 앙상블 (GBM + RF + MLP)
장거리 → CNN-LSTM 3층 (엔진 6개 이상) / 앙상블 (이하)
         ↓
출력: RUL 예측값 (점 추정)
```

### v2 / v12 — 멀티헤드 CNN + 불확실성 + conformal (미실행)
```
입력: 센서 데이터 (regime 정규화 + 잔차, L=50 윈도우)
         ↓
멀티헤드 CNN (Conv1d 3층 → 공유 FC) — fold당 3개 시드 딥앙상블
         ↓
   ┌───────┬───────┬─────────┬───────────┐
  RUL 평균  RUL 분산  고장부위 분류  열화시작   모듈별 열화량(θ)
  (mu)    (lv)      (fault)      (hsh)       (theta)
         ↓
conformal 예측구간 (RUL ± Q, 90% 커버리지 목표)
```

두 구조의 차이는 [`results/v2_8stage_design_note.md`](./results/v2_8stage_design_note.md) 3절 비교표 참고.

---

## 완료된 것 / 진행 중 / 남은 것

**완료**
- v1.0: 전처리 파이프라인 + 앙상블/CNN-LSTM 학습, 실험 A/B/C로 하이퍼파라미터 확정

**진행 중**
- v1.1: DS03-012 중·장거리 실행 + 최종 종합 시각화 필요
- v1.1에서 드러난 "LCC 가중치 방식은 효과가 거의 없다"는 결론을 팀과 공유하고 Stage 2 접근 자체를 재논의

**남은 것**
- v2/v12 8단계 파이프라인 첫 실행 (CAP=50, CAP=125 각각)
- v2 결과가 나오면 v1 대비 RUL 정확도·고장분류 정확도·conformal 커버리지 비교
- DASHlink / NGAFID 데이터로 검증 (FPR ≤ 15%, F1 ≥ 0.80 — 아직 미측정)
- 실제 제주항공 운항 데이터 적용
- 21개 센서 → 8개 공통 센서로 경량화(Knowledge Distillation) — v2에서 아직 다루지 않음, 별도 단계로 남아 있음