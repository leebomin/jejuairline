# ================================================================
# final_baseline.py 결과 전체 저장 코드
# 실험 결과 + 전처리 데이터 + 모델 전부 저장
# 내일 load_final.py 하나만 실행하면 이어서 가능
# ================================================================

import os, pickle, joblib, warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings('ignore')

SAVE_DIR = '/content/drive/MyDrive/제주항공/최종베이스라인저장'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"저장 위치: {SAVE_DIR}\n")


# ── 1. 설정값 저장 ─────────────────────────────────────────────
config = {
    'FOLDER'        : FOLDER,
    'FILES'         : FILES,
    'WINDOW'        : WINDOW,
    'MAX_RUL'       : MAX_RUL,
    'SENSOR_STD_MIN': SENSOR_STD_MIN,
    'FEATURE_RATIO' : FEATURE_RATIO,
    'N_ESTIMATORS'  : N_ESTIMATORS,
    'STEP'          : STEP,
    'MAX_SAMPLES'   : MAX_SAMPLES,
    'LCC_WEIGHT'    : LCC_WEIGHT,
    'CNN_LAYERS'    : CNN_LAYERS,
    'CNN_EPOCHS'    : CNN_EPOCHS,
    'CNN_BATCH_SIZE': CNN_BATCH_SIZE,
    'CNN_LR'        : CNN_LR,
}
with open(f'{SAVE_DIR}/config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("[OK] [1] 설정값 저장 완료")


# ── 2. 실험 결과(RMSE 숫자) 저장 ──────────────────────────────
with open(f'{SAVE_DIR}/results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("[OK] [2] 실험 결과 저장 완료")


# ── 3. 데이터셋별 전처리 데이터 + 모델 저장 ───────────────────
# cross_val 안에서 모델이 만들어지고 사라지기 때문에
# 여기서 다시 전처리 + 학습해서 저장
import h5py
print("\n[3] 데이터셋별 전처리 + 모델 학습 + 저장 시작...")

all_data   = {}   # 전처리 데이터
all_models = {}   # 학습된 모델

for filepath in FILES:
    if not os.path.exists(filepath):
        continue

    ds_key = filepath.split('/')[-1].replace(
        'N-CMAPSS_','').replace('.h5','')
    print(f"\n  {ds_key}")

    with h5py.File(filepath, 'r') as f:
        W  = f['W_dev'][:]
        Xs = f['X_s_dev'][:]
        Y  = f['Y_dev'][:].flatten()
        A  = f['A_dev'][:]
    units = A[:, 0].astype(int)
    fc    = A[:, 2].astype(int)

    for fc_type in np.unique(fc):
        mask  = (fc == fc_type)
        label = f"{ds_key}_{FC_LABELS.get(fc_type, str(fc_type))}"
        n_u   = len(np.unique(units[mask]))

        if n_u < 2:
            continue

        F_sel, Xw, Yw, Uw = preprocess(
            W[mask], Xs[mask], Y[mask], units[mask])
        if F_sel is None:
            continue

        print(f"    {FC_LABELS.get(fc_type,'?')} | 샘플 {len(Yw):,}개", end="  ")

        # 전처리 데이터 저장
        all_data[label] = {
            'F_sel': F_sel,
            'Xw'   : Xw,
            'Yw'   : Yw,
            'Uw'   : Uw,
            'fc'   : fc_type,
            'ds'   : ds_key,
        }

        # ── 앙상블 모델 학습 (전체 데이터로) ──────────────────
        gbm = GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS, max_depth=4, random_state=42)
        rf  = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, max_depth=8, random_state=42)
        mlp = MLPRegressor(
            hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
        gbm.fit(F_sel, Yw)
        rf.fit (F_sel, Yw)
        mlp.fit(F_sel, Yw)

        joblib.dump(gbm, f'{SAVE_DIR}/gbm_{label}.pkl')
        joblib.dump(rf,  f'{SAVE_DIR}/rf_{label}.pkl')
        joblib.dump(mlp, f'{SAVE_DIR}/mlp_{label}.pkl')

        # ── CNN-LSTM 모델 학습 ─────────────────────────────────
        cnn_model = train_cnnlstm(Xw, Yw)
        torch.save(cnn_model.state_dict(),
                   f'{SAVE_DIR}/cnnlstm_{label}.pt')

        # ── LCC 보정 모델 (단거리만) ───────────────────────────
        if fc_type == 1:
            gbm_l, rf_l, mlp_l = lcc_finetune(F_sel, Xw, Yw, Uw, fc[mask])
            joblib.dump(gbm_l, f'{SAVE_DIR}/gbm_lcc_{label}.pkl')
            joblib.dump(rf_l,  f'{SAVE_DIR}/rf_lcc_{label}.pkl')
            joblib.dump(mlp_l, f'{SAVE_DIR}/mlp_lcc_{label}.pkl')
            print("앙상블+CNN+LCC보정 저장 완료")
        else:
            print("앙상블+CNN 저장 완료")

        all_models[label] = {
            'has_lcc': (fc_type == 1),
            'n_sensors': Xw.shape[2],
        }

# 전처리 데이터 통합 저장
with open(f'{SAVE_DIR}/all_data.pkl', 'wb') as f:
    pickle.dump(all_data, f)
with open(f'{SAVE_DIR}/all_models_meta.pkl', 'wb') as f:
    pickle.dump(all_models, f)
print(f"\n[OK] [3] 전체 {len(all_data)}개 케이스 저장 완료")


# ── 4. 저장 목록 출력 ──────────────────────────────────────────
print("\n" + "━"*50)
print("  저장된 파일 목록")
print("━"*50)
total_mb = 0
for fname in sorted(os.listdir(SAVE_DIR)):
    mb = os.path.getsize(f'{SAVE_DIR}/{fname}') / 1e6
    total_mb += mb
    print(f"  {fname:40s}  {mb:6.1f} MB")
print("━"*50)
print(f"  총 크기: {total_mb:.0f} MB")
print("\n저장 완료!")
print("   내일 load_final.py 하나만 실행하면 이어서 시작할 수 있어요.")
