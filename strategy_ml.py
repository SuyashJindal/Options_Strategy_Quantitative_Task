import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, time, date
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
warnings.filterwarnings('ignore')
df_options_data= pd.read_parquet('options_data_2023.parquet')
spot_with_signals = pd.read_csv('spot_with_signals_2023.csv')
mapping = {'Hold': 0, 'Buy': 1, 'Sell': -1}
spot_with_signals['cross'] = spot_with_signals['cross'].fillna(0)
spot_with_signals['signal_encoded'] = spot_with_signals['signal'].map(mapping)
df = spot_with_signals.copy()
H = 3  
fwd_ret = df['close'].pct_change(H).shift(-H)
eps = 0.0003  
y_full = np.where(fwd_ret > eps, 1, np.where(fwd_ret < -eps, 0, np.nan))
base_feats = ['ap','esa','d','ci','tci','wt1','wt2','rsi','cross','signal_encoded',
              'open','high','low','close']
df['ret1'] = df['close'].pct_change(1)
df['ret3'] = df['close'].pct_change(3)
df['hl_range'] = (df['high'] - df['low']) / df['close'].shift(1)
df['body'] = (df['close'] - df['open']) / df['open']
for c in ['ap','esa','d','ci','tci','wt1','wt2','rsi','cross','ret1','ret3','hl_range','body','signal_encoded']:
    df[f'{c}_lag1'] = df[c].shift(1)
roll_cols = ['rsi','wt1','wt2','ret1','hl_range','body']
for c in roll_cols:
    df[f'{c}_r3'] = df[c].shift(1).rolling(3).mean()
    df[f'{c}_r5'] = df[c].shift(1).rolling(5, min_periods=3).mean()  # allows partial windows
    df[f'{c}_s5']  = df[c].shift(1).rolling(5, min_periods=3).std()
df['wt_diff_lag1'] = (df['wt1'] - df['wt2']).shift(1)
tr = np.maximum(df['high']-df['low'],
                np.maximum((df['high']-df['close'].shift(1)).abs(),
                           (df['low']-df['close'].shift(1)).abs()))
df['atr5'] = tr.shift(1).rolling(5).mean()
df['atr5_n'] = df['atr5'] / df['close'].shift(1)
feat_cols = [c for c in df.columns if any(s in c for s in
              ['ap','esa','d','ci','tci','wt','rsi','cross','ret','hl_range','body','lag1','r3','r5','s5','wt_diff','atr5_n','signal_encoded'])
             and not c.endswith(('open','high','low','close'))]
X_full = df[feat_cols]
data = pd.concat([X_full, pd.Series(y_full, name='y')], axis=1).dropna()
X_full = data[feat_cols]
y_full = data['y'].astype(int)
n = len(X_full)
test_size = int(0.2 * n)
val_size = int(0.1 * n)  
X_train_all = X_full.iloc[:n - test_size]
y_train_all = y_full.iloc[:n - test_size]
X_test = X_full.iloc[n - test_size:]
y_test = y_full.iloc[n - test_size:]
X_train = X_train_all.iloc[:-val_size]
y_train = y_train_all.iloc[:-val_size]
X_val = X_train_all.iloc[-val_size:]
y_val = y_train_all.iloc[-val_size:]
for name in ["datetime", "closest_expiry"]:
    for X_ in [X_train, X_val, X_test]:
        if name in X_.columns:
            X_.drop(columns=[name], inplace=True)
drop_cols = ["datetime", "closest_expiry"]
X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns]).copy()
X_val   = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns]).copy()
X_test  = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns]).copy()
X_train = X_train.select_dtypes(include=[np.number])
X_val   = X_val.select_dtypes(include=[np.number])
X_test  = X_test.select_dtypes(include=[np.number])
X_train_ = X_train.astype(float)
X_val_   = X_val.astype(float)
X_test_  = X_test.astype(float)
dtrain = xgb.DMatrix(X_train_, label=y_train)
dval   = xgb.DMatrix(X_val_,   label=y_val)
dtest  = xgb.DMatrix(X_test_,  label=y_test)
pos_ratio = y_train.mean()
neg_ratio = 1 - pos_ratio
scale_pos_weight = (neg_ratio / max(pos_ratio, 1e-6))
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.03,                 # learning_rate
    'max_depth': 5,
    'min_child_weight': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 2.0,               # reg_lambda
    'alpha': 0.0,                # reg_alpha
    'tree_method': 'hist',
    'seed': 42,
    'scale_pos_weight': float(scale_pos_weight),
    'verbosity': 0
}
num_boost_round = 2000
early_stopping_rounds = 150
watchlist = [(dtrain, 'train'), (dval, 'val')]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=watchlist,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=False
)
def predict_with_best(bst, dmat):
    try:
        return bst.predict(dmat, iteration_range=(0, bst.best_iteration + 1))
    except TypeError:
        return bst.predict(dmat, ntree_limit=bst.best_ntree_limit)

p_val  = predict_with_best(bst, dval)
p_test = predict_with_best(bst, dtest)
thr_grid = np.linspace(0.3, 0.7, 81)
best_thr, best_acc = 0.5, -1
for thr in thr_grid:
    acc = accuracy_score(y_val, (p_val >= thr).astype(int))
    if acc > best_acc:
        best_acc, best_thr = acc, thr
y_pred = (p_test >= best_thr).astype(int)
print(f"Chosen threshold on val: {best_thr:.3f} (val acc={best_acc:.4f})")
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
ml_vote = np.where(p_test >= best_thr, 1, -1)
def indicator_vote_row(row):
    votes = []
    if row['rsi'] > 70: votes.append(-1)
    elif row['rsi'] < 30: votes.append(1)
    cross_v = np.sign(row['cross']) if pd.notna(row['cross']) else 0
    votes.append(int(cross_v))
    s = np.sign(np.sum(votes)) if votes else 0
    return int(s if s != 0 else 0)
ind_vote = df.iloc[X_test.index].apply(indicator_vote_row, axis=1).values
signal_core = df.loc[X_test.index, 'signal_encoded'].clip(-1,1).fillna(0).values.astype(int)
comp_score = 0.5*signal_core + 0.3*ml_vote + 0.2*ind_vote
comp_sig = np.where(comp_score >= 0.4, 'Buy', np.where(comp_score <= -0.4, 'Sell', 'Hold'))
out = pd.DataFrame({
    'datetime': df.loc[X_test.index, 'datetime'].values if 'datetime' in df.columns else X_test.index,
    'y_true': y_test.values,
    'proba': p_test,
    'ml_vote': ml_vote,
    'ind_vote': ind_vote,
    'signal_core': signal_core,
    'composite_score': comp_score,
    'composite_signal': comp_sig
})
# 8.1 Predict probabilities for ALL rows in X_full
X_all_ = X_full.select_dtypes(include=[np.number]).astype(float)
dall = xgb.DMatrix(X_all_, label=y_full)
p_all = predict_with_best(bst, dall)   # uses best_iteration
ml_vote_all = np.where(p_all >= best_thr, 1, -1)
def indicator_vote_row(row):
    votes = []
    # RSI (mean-reversion)
    if pd.notna(row.get('rsi', np.nan)):
        if row['rsi'] > 70: votes.append(-1)
        elif row['rsi'] < 30: votes.append(1)
    # CROSS (assumed positive = buy, negative = sell)
    cross_v = row.get('cross', 0)
    if pd.notna(cross_v):
        votes.append(int(np.sign(cross_v)))
    s = np.sign(np.sum(votes)) if votes else 0
    return int(s) if s != 0 else 0
ml_proba_scored = pd.Series(p_all, index=X_full.index)
ml_vote_scored  = pd.Series(ml_vote_all, index=X_full.index)
ml_proba_full = ml_proba_scored.reindex(df.index).fillna(0.5)  
ml_vote_full  = ml_vote_scored.reindex(df.index).fillna(0).astype(int)
indicator_vote_full = df.apply(indicator_vote_row, axis=1).fillna(0).astype(int)
signal_core_full = (
    df.get('signal_encoded', pd.Series(0, index=df.index))
      .clip(-1, 1).fillna(0).astype(int)
)
w_core, w_ml, w_ind = 0.5, 0.3, 0.2
comp_score_full = w_core*signal_core_full + w_ml*ml_vote_full + w_ind*indicator_vote_full
comp_num_full   = np.where(comp_score_full >= 0.4, 1,
                    np.where(comp_score_full <= -0.4, -1, 0))
comp_lbl_full   = pd.Series(comp_num_full, index=df.index).map({1:'Buy', -1:'Sell', 0:'Hold'})
df = df.assign(
    ml_proba=ml_proba_full,
    ml_vote=ml_vote_full,
    indicator_vote=indicator_vote_full,
    signal_core=signal_core_full,
    composite_score=comp_score_full,
    composite_signal_num=comp_num_full,
    composite_signal=comp_lbl_full
)

spot_with_signals['composite_signal'] = df['composite_signal']
df_options_data['expiry_date'] = pd.to_datetime(df_options_data['expiry_date'])
df_options_data['expiry_date']=df_options_data['expiry_date'].dt.date
#print(spot_with_signals)