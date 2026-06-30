# ================================================================
# 최종 베이스라인 코드
# 실험 결과 반영:
#   WINDOW    = 50   (실험 B 최적값)
#   LCC 가중치 = 5배  (실험 A 최적값)
#   CNN 깊이   = 3층  (실험 C 설정)
#   모델       = 앙상블(GBM+RF+MLP) + CNN-LSTM
# ================================================================

# ── 라이브러리 ─────────────────────────────────────────────────
import os, warnings, pickle, joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
print("✅ 라이브러리 로드 완료")

# ================================================================
# ★ 설정값 — 실험 결과 반영
# ================================================================
FOLDER   = '/content/drive/MyDrive/제주항공'
SAVE_DIR = '/content/drive/MyDrive/제주항공/최종베이스라인'
os.makedirs(SAVE_DIR, exist_ok=True)

FILES = [
    f'{FOLDER}/N-CMAPSS_DS01-005.h5',
    f'{FOLDER}/N-CMAPSS_DS02-006.h5',
    f'{FOLDER}/N-CMAPSS_DS03-012.h5',
]

# 실험으로 확정된 최적 파라미터
WINDOW         = 50      # ★ 실험 B: 50이 최적
MAX_RUL        = 125
SENSOR_STD_MIN = 0.01
FEATURE_RATIO  = 0.7
N_ESTIMATORS   = 150
STEP           = 10
MAX_SAMPLES    = 30000
LCC_WEIGHT     = 5       # ★ 실험 A: 5배가 최적

# CNN-LSTM 설정
CNN_LAYERS     = 3       # ★ 3층 고정
CNN_EPOCHS     = 20
CNN_BATCH_SIZE = 256
CNN_LR         = 1e-3

FC_LABELS = {1: '단거리', 2: '중거리', 3: '장거리'}
FC_COLORS = {1: '#E87A5D', 2: '#F5B942', 3: '#4C8EDA'}

print(f"\n설정값 확인:")
print(f"  WINDOW      = {WINDOW}   (실험 B 최적)")
print(f"  LCC 가중치  = {LCC_WEIGHT}배   (실험 A 최적)")
print(f"  CNN 층수    = {CNN_LAYERS}층   (실험 C 설정)")
print(f"  모델        = 앙상블 + CNN-LSTM")


# ================================================================
# CNN-LSTM 3층 모델
# ================================================================
class CNNLSTM(nn.Module):
    """
    CNN 3층 + LSTM
    1층: 단순 패턴 (온도 변화)
    2층: 복합 패턴 (여러 센서 동시 변화)
    3층: 추상 패턴 (노화 신호 종합)
    """
    def __init__(self, n_sensors, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(n_sensors, 32,  3, padding=1)
        self.conv2 = nn.Conv1d(32,        64,  3, padding=1)
        self.conv3 = nn.Conv1d(64,        128, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.bn2   = nn.BatchNorm1d(64)
        self.bn3   = nn.BatchNorm1d(128)
        self.lstm  = nn.LSTM(128, hidden, 2,
                             batch_first=True,
                             bidirectional=True,
                             dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


def train_cnnlstm(Xw_tr, Y_tr):
    n_sensors = Xw_tr.shape[2]
    model     = CNNLSTM(n_sensors=n_sensors)
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR)
    criterion = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(Xw_tr),
        torch.FloatTensor(Y_tr)
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=CNN_BATCH_SIZE, shuffle=True)
    model.train()
    for epoch in range(CNN_EPOCHS):
        for xb, yb in dl:
            optimizer.zero_grad()
            nn.MSELoss()(model(xb), yb).backward()
            optimizer.step()
    return model


def predict_cnnlstm(model, Xw):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(Xw)).numpy()


# ================================================================
# 공통 함수
# ================================================================
def load_file(filepath):
    print(f"  읽는 중...", end=" ")
    with h5py.File(filepath, 'r') as f:
        W  = f['W_dev'][:]
        Xs = f['X_s_dev'][:]
        Y  = f['Y_dev'][:].flatten()
        A  = f['A_dev'][:]
    units = A[:, 0].astype(int)
    fc    = A[:, 2].astype(int)
    print(f"완료 ({len(Y):,}개 시점)")
    return W, Xs, Y, units, fc


def preprocess(W, Xs, Y, units):
    """WINDOW=50 고정 전처리"""
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

    Xw = np.array(Xw)
    Yw = np.array(Yw)
    Uw = np.array(Uw)

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


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cross_val(F, Xw, Y, units, model_type='ensemble'):
    """유닛 교차검증"""
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
            gbm.fit(F_tr, Y_tr)
            rf.fit (F_tr, Y_tr)
            mlp.fit(F_tr, Y_tr)
            pred = (gbm.predict(F_te) +
                    rf.predict(F_te)  +
                    mlp.predict(F_te)) / 3

        elif model_type == 'cnnlstm':
            model = train_cnnlstm(Xw[tr], Y_tr)
            pred  = predict_cnnlstm(model, Xw[te])

        fold_rmses.append(rmse(Y_te, pred))

    if not fold_rmses:
        return None, []
    return float(np.mean(fold_rmses)), fold_rmses


def lcc_finetune(F, Xw, Y, units, fc, model_type='ensemble'):
    """
    LCC few-shot fine-tuning
    전체 데이터의 20%를 LCC_WEIGHT배 가중치로 추가해서 재학습
    """
    n_few = int(len(F) * 0.2)
    F_combined = np.vstack([F, F[:n_few]])
    Y_combined = np.concatenate([Y, Y[:n_few]])
    weights    = np.concatenate([
        np.ones(len(F)),
        np.ones(n_few) * LCC_WEIGHT   # 5배 가중치
    ])
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


# ================================================================
# 실험 실행
# ================================================================
results = {}   # 전체 결과 저장

for filepath in FILES:
    if not os.path.exists(filepath):
        print(f"\n⚠️  {filepath.split('/')[-1]} 없음, 건너뜀")
        continue

    ds_key = filepath.split('/')[-1].replace(
        'N-CMAPSS_', '').replace('.h5', '')

    print(f"\n{'═'*55}")
    print(f"  📂  {ds_key}")
    print(f"{'═'*55}")

    W, Xs, Y, units, fc = load_file(filepath)
    fc_types = np.unique(fc)
    print(f"  비행 클래스: {[FC_LABELS.get(f, str(f)) for f in fc_types]}")

    for fc_type in fc_types:
        mask   = (fc == fc_type)
        label  = f"{ds_key} / {FC_LABELS.get(fc_type, str(fc_type))}"
        n_u    = len(np.unique(units[mask]))

        print(f"\n  ▶ {FC_LABELS.get(fc_type,'?')} | 엔진 {n_u}개", end="  ")
        if n_u < 2:
            print("→ ⚠️ 엔진 부족"); continue

        F_sel, Xw, Yw, Uw = preprocess(
            W[mask], Xs[mask], Y[mask], units[mask])
        if F_sel is None:
            print("→ ⚠️ 데이터 부족"); continue

        print(f"| 샘플 {len(Yw):,}개")

        # ── 앙상블 (교차검증) ───────────────────────────────────
        print(f"    앙상블    학습 중...", end="  ")
        r_ens, folds_ens = cross_val(F_sel, Xw, Yw, Uw, 'ensemble')
        if r_ens:
            fold_str = [f'{v:.1f}' for v in folds_ens]
            print(f"→ RMSE: {r_ens:.2f}  폴드별: {fold_str}")
        else:
            print("→ 실패")

        # ── CNN-LSTM 3층 (교차검증) ─────────────────────────────
        print(f"    CNN-LSTM  학습 중...", end="  ")
        r_cnn, folds_cnn = cross_val(F_sel, Xw, Yw, Uw, 'cnnlstm')
        if r_cnn:
            fold_str = [f'{v:.1f}' for v in folds_cnn]
            print(f"→ RMSE: {r_cnn:.2f}  폴드별: {fold_str}")
        else:
            print("→ 실패")

        # ── LCC 가중치 5배 앙상블 (단거리만) ───────────────────
        lcc_r = None
        if fc_type == 1:
            print(f"    LCC보정   학습 중...", end="  ")
            gbm_l, rf_l, mlp_l = lcc_finetune(
                F_sel, Xw, Yw, Uw, fc[mask])
            if gbm_l:
                # 마지막 엔진으로 평가
                te_u  = np.unique(Uw)[-1]
                te    = (Uw == te_u)
                pred  = (gbm_l.predict(F_sel[te]) +
                         rf_l.predict(F_sel[te])  +
                         mlp_l.predict(F_sel[te])) / 3
                lcc_r = rmse(Yw[te], pred)
                print(f"→ RMSE: {lcc_r:.2f}  (LCC 가중치 {LCC_WEIGHT}배)")
            else:
                print("→ 실패")

        results[label] = {
            'rmse_ensemble': r_ens,
            'rmse_cnnlstm' : r_cnn,
            'rmse_lcc'     : lcc_r,
            'folds_ens'    : folds_ens,
            'folds_cnn'    : folds_cnn,
            'fc'           : fc_type,
            'ds'           : ds_key,
        }


# ================================================================
# 시각화
# ================================================================
print("\n\n📊 그래프 그리는 중...")

labels = list(results.keys())
n      = len(labels)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── 왼쪽: 앙상블 vs CNN-LSTM RMSE 비교 ────────────────────────
ax = axes[0]
x       = np.arange(n)
width   = 0.35
rmse_e  = [results[k]['rmse_ensemble'] or 0 for k in labels]
rmse_c  = [results[k]['rmse_cnnlstm']  or 0 for k in labels]

bars1 = ax.bar(x - width/2, rmse_e, width,
               label='앙상블', color='#4C8EDA', edgecolor='white')
bars2 = ax.bar(x + width/2, rmse_c, width,
               label='CNN-LSTM 3층', color='#E87A5D', edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels([l.replace(' / ','\n') for l in labels], fontsize=8)
ax.set_ylabel('RMSE  (낮을수록 정확)')
ax.set_title(f'앙상블 vs CNN-LSTM 비교\n(WINDOW={WINDOW}, 유닛 교차검증)',
             fontweight='bold')
ax.legend()

for bar, val in zip(bars1, rmse_e):
    if val:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)
for bar, val in zip(bars2, rmse_c):
    if val:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)

# ── 오른쪽: 단거리 LCC 보정 효과 ──────────────────────────────
ax = axes[1]
short_labels = [k for k in labels if results[k]['fc'] == 1]

if short_labels:
    x2      = np.arange(len(short_labels))
    width2  = 0.25
    base_e  = [results[k]['rmse_ensemble'] or 0 for k in short_labels]
    base_c  = [results[k]['rmse_cnnlstm']  or 0 for k in short_labels]
    lcc_v   = [results[k]['rmse_lcc']      or 0 for k in short_labels]

    ax.bar(x2 - width2, base_e, width2,
           label='앙상블(보정 전)', color='#4C8EDA', alpha=0.6, edgecolor='white')
    ax.bar(x2,           lcc_v,  width2,
           label=f'앙상블(LCC {LCC_WEIGHT}배)', color='#4C8EDA', edgecolor='white')
    ax.bar(x2 + width2, base_c, width2,
           label='CNN-LSTM', color='#E87A5D', edgecolor='white')

    ax.set_xticks(x2)
    ax.set_xticklabels(
        [l.replace(' / ','\n') for l in short_labels], fontsize=9)
    ax.set_ylabel('RMSE')
    ax.set_title(f'단거리 LCC 가중치 {LCC_WEIGHT}배 보정 효과',
                 fontweight='bold')
    ax.legend(fontsize=9)
else:
    ax.text(0.5, 0.5, '단거리 데이터 없음',
            ha='center', va='center', transform=ax.transAxes)

plt.suptitle(f'최종 베이스라인  |  WINDOW={WINDOW} / LCC가중치={LCC_WEIGHT}배 / CNN {CNN_LAYERS}층',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/최종베이스라인.png', dpi=150, bbox_inches='tight')
plt.show()


# ================================================================
# 요약 출력
# ================================================================
print("\n" + "━"*55)
print("  📌 최종 베이스라인 요약")
print(f"     WINDOW={WINDOW} / LCC가중치={LCC_WEIGHT}배 / CNN {CNN_LAYERS}층")
print("━"*55)

print(f"\n  {'실험':30s}  {'앙상블':>7}  {'CNN-LSTM':>9}  {'LCC보정':>7}")
print("  " + "-"*55)
for k, v in sorted(results.items(),
                    key=lambda x: x[1]['rmse_ensemble'] or 99):
    e = f"{v['rmse_ensemble']:.2f}" if v['rmse_ensemble'] else "  -  "
    c = f"{v['rmse_cnnlstm']:.2f}"  if v['rmse_cnnlstm']  else "  -  "
    l = f"{v['rmse_lcc']:.2f}"      if v['rmse_lcc']      else "  -  "
    print(f"  {k:30s}  {e:>7}  {c:>9}  {l:>7}")

print("━"*55)

# 전체 평균
e_vals = [v['rmse_ensemble'] for v in results.values() if v['rmse_ensemble']]
c_vals = [v['rmse_cnnlstm']  for v in results.values() if v['rmse_cnnlstm']]
if e_vals and c_vals:
    print(f"\n  앙상블 전체 평균 RMSE:    {np.mean(e_vals):.2f}")
    print(f"  CNN-LSTM 전체 평균 RMSE:  {np.mean(c_vals):.2f}")
    if np.mean(e_vals) < np.mean(c_vals):
        print(f"\n  ✅ 앙상블이 CNN-LSTM보다 평균 "
              f"{np.mean(c_vals)-np.mean(e_vals):.2f} 낮아요.")
    else:
        print(f"\n  ✅ CNN-LSTM이 앙상블보다 평균 "
              f"{np.mean(e_vals)-np.mean(c_vals):.2f} 낮아요.")

# ================================================================
# 저장
# ================================================================
with open(f'{SAVE_DIR}/results.pkl', 'wb') as f:
    pickle.dump(results, f)

config = {
    'WINDOW': WINDOW, 'MAX_RUL': MAX_RUL,
    'LCC_WEIGHT': LCC_WEIGHT, 'CNN_LAYERS': CNN_LAYERS,
    'N_ESTIMATORS': N_ESTIMATORS, 'STEP': STEP,
    'MAX_SAMPLES': MAX_SAMPLES,
}
with open(f'{SAVE_DIR}/config.pkl', 'wb') as f:
    pickle.dump(config, f)

print(f"\n💾 결과 저장 완료: {SAVE_DIR}")
print("━"*55)
