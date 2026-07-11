# CHANGELOG

날짜는 코드 파일명에 쓰인 작업일 기준(YYMMDD).

---

## v2 / 내부 코드명 "v12" — 8단계 멀티태스크 파이프라인 — 2026-07-01

**상태: 🟡 미실행 (코드만 작성, 기록용 업로드)**

- 파일: `notebooks/260701_8stage_cap50.ipynb`, `notebooks/260701_8stage_cap125.ipynb`
- 앙상블+CNN-LSTM을 따로 비교하던 구조를 버리고, **멀티헤드 CNN 하나**가 RUL·고장부위·열화시작·건강지표(θ)를 동시에 예측하는 구조로 전면 개편
- 불확실성 정량화(heteroscedastic NLL) + 딥앙상블 + conformal 예측구간 추가
- fold/모델 단위 체크포인트로 세션이 끊겨도 이어서 실행 가능
- CAP(RUL 상한) 50 버전과 125 버전을 별도 파일로 분리 (체크포인트 폴더 분리, 결과 직접 비교 금지)
- 설계 상세: [`results/v2_8stage_design_note.md`](./results/v2_8stage_design_note.md)

## v1.1 — 피드백 반영 · 버그 수정 — 2026-06-30

**상태: 🟡 부분 실행 (DS03-012 중·장거리, 최종 종합 시각화 미실행)**

- 파일: `notebooks/260630-v1_1.ipynb` (v1.0의 `src/final_baseline.py`를 대체)
- **[F1]** LCC 가중치 효과를 1/3/5/10배로 스윕 재검증 → 데이터 누수 제거 후에는 개선 효과가 사실상 사라짐 (v1.0이 보고한 15~20% 개선은 대부분 누수로 인한 착시였음)
- **[F2]** `cross_val()`이 평균과 함께 표준편차·엔진 수를 반환 (단거리 결과의 큰 분산이 모델 결함이 아니라 엔진 수 부족 때문임을 명시)
- **[F3]** GPU 사용 + `MAX_SAMPLES` 제한 해제
- **[F4]** `sample_weight`를 지원하지 않는 MLP를 가중 학습 대상에서 제외 (GBM+RF만 가중 학습)
- **[F5]** 스케일러·특징선택을 전체 데이터가 아닌 **교차검증 train 폴드 내부에서만** fit하도록 수정 (데이터 누수 근본 원인 제거)
- 그 외 재현성 시드 고정, GPU 배치 추론, BatchNorm 마지막 배치 크래시 등 버그 5건(B1–B5) 수정
- 결과 상세: [`results/v1.1_bugfix_and_results.md`](./results/v1.1_bugfix_and_results.md)

## v1.0 — 최초 베이스라인 — 2026-06-26 ~ 06-29

**상태: 🟢 완료**

- 파일: `src/final_baseline.py`, `src/save_final.py`, `src/load_final.py` (원본 실험 노트북 `260630.ipynb` 기반)
- WINDOW=50, LCC 가중치=5배, CNN-LSTM 3층으로 확정한 최초 베이스라인
- 실험 A(LCC 가중치)/B(WINDOW 크기)/C(모델 비교)로 하이퍼파라미터 확정
- 결과 상세: [`results/results_summary.md`](./results/results_summary.md), [`results/team_comparison.md`](./results/team_comparison.md)
- ⚠️ 이 버전에서 보고된 LCC 개선 효과(약 15%)는 v1.1에서 데이터 누수로 밝혀졌다. 수치를 인용할 때는 v1.1 문서를 함께 참고할 것.
