# ================================================================
# load_final.py — 다음날 이것만 실행하면 됩니다
# final_baseline.py 전체를 다시 돌리지 않아도
# 모든 변수, 모델, 데이터가 복원됩니다
# ================================================================

import os, pickle, joblib, warnings
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')

SAVE_DIR = '/content/drive/MyDrive/제주항공/최종베이스라인저장'


# ── 1. 설정값 복원 ─────────────────────────────────────────────
with open(f'{SAVE_DIR}/config.pkl', 'rb') as f:
    config = pickle.load(f)

FOLDER         = config['FOLDER']
FILES          = config['FILES']
WINDOW         = config['WINDOW']
MAX_RUL        = config['MAX_RUL']
SENSOR_STD_MIN = config['SENSOR_STD_MIN']
FEATURE_RATIO  = config['FEATURE_RATIO']
N_ESTIMATORS   = config['N_ESTIMATORS']
STEP           = config['STEP']
MAX_SAMPLES    = config['MAX_SAMPLES']
LCC_WEIGHT     = config['LCC_WEIGHT']
CNN_LAYERS     = config['CNN_LAYERS']
CNN_EPOCHS     = config['CNN_EPOCHS']
CNN_BATCH_SIZE = config['CNN_BATCH_SIZE']
CNN_LR         = config['CNN_LR']

FC_LABELS = {1: '단거리', 2: '중거리', 3: '장거리'}
FC_COLORS = {1: '#E87A5D', 2: '#F5B942', 3: '#4C8EDA'}
print("✅ [1] 설정값 복원 완료")


# ── 2. 실험 결과 복원 ──────────────────────────────────────────
with open(f'{SAVE_DIR}/results.pkl', 'rb') as f:
    results = pickle.load(f)
print(f"✅ [2] 실험 결과 복원 완료  ({len(results)}개 케이스)")


# ── 3. 전처리 데이터 복원 ──────────────────────────────────────
with open(f'{SAVE_DIR}/all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)
with open(f'{SAVE_DIR}/all_models_meta.pkl', 'rb') as f:
    all_models_meta = pickle.load(f)
print(f"✅ [3] 전처리 데이터 복원 완료  ({len(all_data)}개 케이스)")


# ── 4. CNN-LSTM 클래스 정의 (모델 로드에 필요) ─────────────────
class CNNLSTM(nn.Module):
    def __init__(self, n_sensors, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(n_sensors, 32,  3, padding=1)
        self.conv2 = nn.Conv1d(32,        64,  3, padding=1)
        self.conv3 = nn.Conv1d(64,        128, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.bn2   = nn.BatchNorm1d(64)
        self.bn3   = nn.BatchNorm1d(128)
        self.lstm  = nn.LSTM(128, hidden, 2, batch_first=True,
                             bidirectional=True, dropout=0.2)
        self.fc    = nn.Sequential(
            nn.Linear(hidden*2, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 1)
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


# ── 5. 모델 복원 ───────────────────────────────────────────────
loaded_models = {}

for label, meta in all_models_meta.items():
    loaded_models[label] = {}

    # 앙상블
    loaded_models[label]['gbm'] = joblib.load(
        f'{SAVE_DIR}/gbm_{label}.pkl')
    loaded_models[label]['rf']  = joblib.load(
        f'{SAVE_DIR}/rf_{label}.pkl')
    loaded_models[label]['mlp'] = joblib.load(
        f'{SAVE_DIR}/mlp_{label}.pkl')

    # CNN-LSTM
    cnn = CNNLSTM(n_sensors=meta['n_sensors'])
    cnn.load_state_dict(
        torch.load(f'{SAVE_DIR}/cnnlstm_{label}.pt',
                   map_location='cpu'))
    cnn.eval()
    loaded_models[label]['cnnlstm'] = cnn

    # LCC 보정 모델 (단거리만)
    if meta['has_lcc']:
        loaded_models[label]['gbm_lcc'] = joblib.load(
            f'{SAVE_DIR}/gbm_lcc_{label}.pkl')
        loaded_models[label]['rf_lcc']  = joblib.load(
            f'{SAVE_DIR}/rf_lcc_{label}.pkl')
        loaded_models[label]['mlp_lcc'] = joblib.load(
            f'{SAVE_DIR}/mlp_lcc_{label}.pkl')

print(f"✅ [4] 모델 복원 완료")
for label in loaded_models:
    has_lcc = '(LCC보정 포함)' if all_models_meta[label]['has_lcc'] else ''
    print(f"     {label}  앙상블+CNN-LSTM {has_lcc}")


# ── 6. 공통 함수 정의 ──────────────────────────────────────────
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def preprocess(W, Xs, Y, units):
    X_raw  = np.concatenate([W, Xs], axis=1)
    valid  = X_raw.std(axis=0) > SENSOR_STD_MIN
    X_raw  = X_raw[:, valid]
    X_norm = MinMaxScaler().fit_transform(X_raw)
    Y_clip = np.clip(Y, 0, MAX_RUL)
    Xw, Yw, Uw = [], [], []
    for u in np.unique(units):
        mask = (units == u)
        Xu   = X_norm[mask][::STEP]
        Yu   = Y_clip[mask][::STEP]
        if len(Xu) <= WINDOW:
            continue
        for i in range(len(Xu) - WINDOW):
            Xw.append(Xu[i:i+WINDOW])
            Yw.append(Yu[i+WINDOW])
            Uw.append(u)
    if len(Yw) == 0:
        return None, None, None, None
    Xw = np.array(Xw); Yw = np.array(Yw); Uw = np.array(Uw)
    if len(Yw) > MAX_SAMPLES:
        idx = np.random.RandomState(42).choice(
            len(Yw), MAX_SAMPLES, replace=False)
        idx.sort()
        Xw, Yw, Uw = Xw[idx], Yw[idx], Uw[idx]
    F = np.concatenate([
        Xw.mean(axis=1), Xw.std(axis=1),
        Xw.max(axis=1),  Xw.min(axis=1),
        Xw[:,-1,:] - Xw[:,0,:],
        np.diff(Xw, axis=1).mean(axis=1),
    ], axis=1)
    k     = max(1, int(F.shape[1] * FEATURE_RATIO))
    F_sel = SelectKBest(f_regression, k=k).fit_transform(F, Yw)
    return F_sel, Xw, Yw, Uw

def predict_ensemble(label, F):
    m = loaded_models[label]
    return (m['gbm'].predict(F) +
            m['rf'].predict(F)  +
            m['mlp'].predict(F)) / 3

def predict_cnnlstm(label, Xw):
    model = loaded_models[label]['cnnlstm']
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(Xw)).numpy()

def predict_lcc(label, F):
    m = loaded_models[label]
    return (m['gbm_lcc'].predict(F) +
            m['rf_lcc'].predict(F)  +
            m['mlp_lcc'].predict(F)) / 3

def train_cnnlstm(Xw_tr, Y_tr):
    n_sensors = Xw_tr.shape[2]
    model     = CNNLSTM(n_sensors=n_sensors)
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR)
    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(Xw_tr), torch.FloatTensor(Y_tr))
    dl = torch.utils.data.DataLoader(
        ds, batch_size=CNN_BATCH_SIZE, shuffle=True)
    model.train()
    for _ in range(CNN_EPOCHS):
        for xb, yb in dl:
            optimizer.zero_grad()
            nn.MSELoss()(model(xb), yb).backward()
            optimizer.step()
    return model

def lcc_finetune(F, Xw, Y, units, fc):
    n_few      = int(len(F) * 0.2)
    F_combined = np.vstack([F, F[:n_few]])
    Y_combined = np.concatenate([Y, Y[:n_few]])
    weights    = np.concatenate([
        np.ones(len(F)), np.ones(n_few) * LCC_WEIGHT])
    gbm = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS, max_depth=4, random_state=42)
    rf  = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=8, random_state=42)
    mlp = MLPRegressor(
        hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
    gbm.fit(F_combined, Y_combined, sample_weight=weights)
    rf.fit (F_combined, Y_combined, sample_weight=weights)
    mlp.fit(F_combined, Y_combined)
    return gbm, rf, mlp

def cross_val(F, Xw, Y, units, model_type='ensemble'):
    all_units = np.unique(units)
    if len(all_units) < 2:
        return None, []
    fold_rmses = []
    for test_unit in all_units:
        tr = (units != test_unit)
        te = (units == test_unit)
        if tr.sum() == 0 or te.sum() == 0:
            continue
        F_tr, Y_tr = F[tr], Y[tr]
        F_te, Y_te = F[te], Y[te]
        if model_type == 'ensemble':
            gbm = GradientBoostingRegressor(
                n_estimators=N_ESTIMATORS, max_depth=4, random_state=42)
            rf  = RandomForestRegressor(
                n_estimators=N_ESTIMATORS, max_depth=8, random_state=42)
            mlp = MLPRegressor(
                hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
            gbm.fit(F_tr, Y_tr); rf.fit(F_tr, Y_tr); mlp.fit(F_tr, Y_tr)
            pred = (gbm.predict(F_te)+rf.predict(F_te)+mlp.predict(F_te))/3
        elif model_type == 'cnnlstm':
            model = train_cnnlstm(Xw[tr], Y_tr)
            model.eval()
            with torch.no_grad():
                pred = model(torch.FloatTensor(Xw[te])).numpy()
        fold_rmses.append(rmse(Y_te, pred))
    if not fold_rmses:
        return None, []
    return float(np.mean(fold_rmses)), fold_rmses

print("✅ [5] 공통 함수 정의 완료")


# ── 7. 원본 데이터 복원 (다음 실험에 필요) ─────────────────────
# 가장 마지막 파일(DS03)을 기본으로 복원
print("\n  원본 데이터 복원 중...", end=" ")
last_file = [f for f in FILES if os.path.exists(f)][-1]
with h5py.File(last_file, 'r') as f:
    W_g  = f['W_dev'][:]
    Xs_g = f['X_s_dev'][:]
    Y_g  = f['Y_dev'][:].flatten()
    A_g  = f['A_dev'][:]
units_g = A_g[:, 0].astype(int)
fc_g    = A_g[:, 2].astype(int)
ds_key  = last_file.split('/')[-1].replace(
    'N-CMAPSS_','').replace('.h5','')
print(f"완료  ({ds_key}, {len(Y_g):,}개 시점)")
print("✅ [6] 원본 데이터 복원 완료")


# ── 복원 완료 요약 ─────────────────────────────────────────────
print("\n" + "━"*55)
print("  🎉 전체 복원 완료!")
print("━"*55)

print(f"\n  복원된 변수:")
print(f"    results          → {len(results)}개 실험 결과 (RMSE)")
print(f"    all_data         → {len(all_data)}개 케이스 전처리 데이터")
print(f"    loaded_models    → {len(loaded_models)}개 케이스 모델 전부")

print(f"\n  사용 가능한 모델:")
for label in loaded_models:
    has_lcc = ' + LCC보정' if all_models_meta[label]['has_lcc'] else ''
    print(f"    {label:30s}  앙상블 + CNN-LSTM{has_lcc}")

print(f"\n  바로 예측 예시:")
first_label = list(loaded_models.keys())[0]
first_data  = all_data[first_label]
print(f"    label = '{first_label}'")
print(f"    F     = all_data[label]['F_sel']")
print(f"    pred  = predict_ensemble(label, F)  # 앙상블 예측")
print(f"    pred  = predict_cnnlstm(label, all_data[label]['Xw'])  # CNN 예측")

print(f"\n  ▶ 이제 다음 실험을 시작하세요!")
print("━"*55)
