import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, 'src')
from features import CAT_COLS

train = pd.read_parquet('data/train_features.parquet')
test = pd.read_parquet('data/test_features.parquet')
orig = pd.read_csv('data/train.csv')
target = (orig['Will_Buy_EV'] == 'Yes').astype(int)

feat_cols = [c for c in train.columns if c != 'id']
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# ---------- XGBoost ----------
oof_xgb = np.zeros(len(train))
test_xgb = np.zeros(len(test))
for c in CAT_COLS:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')

xgb_params = dict(
    objective='binary:logistic', eval_metric='auc', learning_rate=0.03,
    max_depth=7, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, tree_method='hist', enable_categorical=True, device='cpu',
)
for fold, (tr_idx, va_idx) in enumerate(skf.split(train[feat_cols], target)):
    X_tr, y_tr = train.iloc[tr_idx][feat_cols], target.iloc[tr_idx]
    X_va, y_va = train.iloc[va_idx][feat_cols], target.iloc[va_idx]
    dtr = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
    dva = xgb.DMatrix(X_va, y_va, enable_categorical=True)
    dtest = xgb.DMatrix(test[feat_cols], enable_categorical=True)
    model = xgb.train(xgb_params, dtr, num_boost_round=3000, evals=[(dva, 'val')],
                       early_stopping_rounds=100, verbose_eval=False)
    oof_xgb[va_idx] = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
    test_xgb += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / n_splits
    print(f'XGB Fold {fold}: AUC={roc_auc_score(y_va, oof_xgb[va_idx]):.5f}')

print(f'\nXGBoost OOF AUC: {roc_auc_score(target, oof_xgb):.5f}')
np.save('models/stage2_xgb_oof.npy', oof_xgb)
np.save('models/stage2_xgb_test.npy', test_xgb)

# ---------- CatBoost ----------
oof_cat = np.zeros(len(train))
test_cat = np.zeros(len(test))
cat_idx = [feat_cols.index(c) for c in CAT_COLS]
train_cb = train.copy()
test_cb = test.copy()
for c in CAT_COLS:
    train_cb[c] = train_cb[c].astype(str)
    test_cb[c] = test_cb[c].astype(str)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_cb[feat_cols], target)):
    X_tr, y_tr = train_cb.iloc[tr_idx][feat_cols], target.iloc[tr_idx]
    X_va, y_va = train_cb.iloc[va_idx][feat_cols], target.iloc[va_idx]
    model = CatBoostClassifier(
        iterations=3000, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
        eval_metric='AUC', loss_function='Logloss', cat_features=cat_idx,
        early_stopping_rounds=100, verbose=False, random_seed=42, thread_count=-1,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    oof_cat[va_idx] = model.predict_proba(X_va)[:, 1]
    test_cat += model.predict_proba(test_cb[feat_cols])[:, 1] / n_splits
    print(f'CatBoost Fold {fold}: AUC={roc_auc_score(y_va, oof_cat[va_idx]):.5f}')

print(f'\nCatBoost OOF AUC: {roc_auc_score(target, oof_cat):.5f}')
np.save('models/stage2_cat_oof.npy', oof_cat)
np.save('models/stage2_cat_test.npy', test_cat)
