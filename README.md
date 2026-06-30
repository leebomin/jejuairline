# 제주항공 B737 예지정비 AI PoC

미래내일 일경험 프로젝트형 — 제주항공 협업  
수행 기간: 2026년 6월  

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

## 데이터셋

NASA N-CMAPSS (New Commercial Modular Aero-Propulsion System Simulation)

| 파일 | 엔진 수 | 비행 클래스 | 다운로드 |
|-----|--------|-----------|--------|
| DS01-005.h5 | 5개 | 단/중/장거리 | [figshare](https://figshare.com/articles/dataset/N-CMAPSS_DS01-005_h5/14862577) |
| DS02-006.h5 | 6개 | 장거리 | [figshare](https://figshare.com/articles/dataset/N-CMAPSS_DS02-006_h5/14862580) |
| DS03-012.h5 | 12개 | 단/중/장거리 | [figshare](https://figshare.com/articles/dataset/N-CMAPSS_DS03-012_h5/14862586) |

> ⚠️ h5 파일은 용량이 커서 (파일당 약 3GB) 이 저장소에 포함되지 않습니다.  
> 위 링크에서 직접 다운로드 후 Google Drive에 업로드해서 사용하세요.

---

## 핵심 파라미터

| 파라미터 | 값 | 결정 근거 |
|---------|---|---------|
| WINDOW | 50 | 실험 B 비교 결과 최적 |
| MAX_RUL | 125 | 논문 표준값 |
| LCC 가중치 | 5배 | 실험 A 비교 결과 최적 |
| CNN 층수 | 3층 | 실험 C 설정 |
| 특징 선택 | 상위 70% | Alomari 2023 논문 기준 |

---

## 실험 결과

### 최종 RMSE (유닛 교차검증 기준)

| 데이터셋 | 비행 클래스 | 앙상블 | CNN-LSTM | LCC보정 |
|---------|-----------|-------|---------|--------|
| DS01-005 | 단거리 | 14.32 | 13.37 | **12.76** |
| DS01-005 | 중거리 | 16.97 | 27.02 | - |
| DS01-005 | 장거리 | 15.65 | 20.32 | - |
| DS02-006 | 장거리 | 11.27 | **9.57** | - |
| DS03-012 | 단거리 | 14.99 | 15.04 | **12.02** |
| DS03-012 | 중거리 | **12.81** | 15.03 | - |
| DS03-012 | 장거리 | **11.23** | 20.95 | - |

### 매뉴얼 목표 대비

| 항목 | 목표 | 달성 여부 |
|-----|-----|---------|
| RMSE ≤ 9.08 | 9.08 | 장거리 9.57 (근접) / 단거리 12.02 (미달) |
| LCC 개선율 | 30% | 평균 15% (절반 수준) |
| DASHlink FPR | ≤ 15% | 미측정 |
| NGAFID F1 | ≥ 0.80 | 미측정 |

---

## 모델 구조

```
입력: 센서 데이터 (WINDOW=50 슬라이딩 윈도우)
         ↓
[비행 클래스 판별]
         ↓
단거리 → LCC 보정 앙상블 (GBM + RF + MLP, 가중치 5배)
중거리 → 앙상블 (GBM + RF + MLP)
장거리 → CNN-LSTM 3층 (엔진 6개 이상) / 앙상블 (이하)
         ↓
출력: RUL 예측값
```

---

## 파일 설명

```
src/
  final_baseline.py   최종 실험 코드 (WINDOW=50, LCC=5배, CNN 3층)
  save_final.py       실험 결과 + 모델 전체 저장
  load_final.py       다음날 불러오기 (전체 복원)

results/
  results_summary.md  실험 결과 종합 보고서
  team_comparison.md  매뉴얼 대비 비교
```

---

## 실행 방법

```python
# 1. Google Colab에서 final_baseline.py 전체 복붙 후 실행
# 2. FOLDER 경로만 수정
FOLDER = '/content/drive/MyDrive/제주항공'  # 본인 경로로 변경

# 3. 결과 저장
# save_final.py 실행

# 4. 다음날 이어하기
# load_final.py 실행 → 모든 모델/데이터 복원됨
```

---

## 완료된 것 / 남은 것

**완료**
- Stage 1: 전처리 파이프라인 + 앙상블/CNN-LSTM 학습
- Stage 2: LCC few-shot fine-tuning (기본 적용)
- 실험 A/B/C: 파라미터 비교 실험

**남은 것**
- Stage 3: Knowledge Distillation (21→8센서 경량화)
- DASHlink / NGAFID 검증
- 실제 제주항공 운항 데이터 적용
