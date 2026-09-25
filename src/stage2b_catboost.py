import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys, time
sys.path.insert(0, 'src')
from features import CAT_COLS

train = pd.read_parquet('data/train_features.parquet')
test = pd.read_parquet('data/test_features.parquet')
orig = pd.read_csv('data/train.csv')
target = (orig['Will_Buy_EV'] == 'Yes').astype(int)

feat_cols = [c for c in train.columns if c != 'id']
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_cat = np.zeros(len(train))
test_cat = np.zeros(len(test))
cat_idx = [feat_cols.index(c) for c in CAT_COLS]
for c in CAT_COLS:
    train[c] = train[c].astype(str)
    test[c] = test[c].astype(str)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train[feat_cols], target)):
    t0 = time.time()
    X_tr, y_tr = train.iloc[tr_idx][feat_cols], target.iloc[tr_idx]
    X_va, y_va = train.iloc[va_idx][feat_cols], target.iloc[va_idx]
    model = CatBoostClassifier(
        iterations=1500, learning_rate=0.06, depth=6, l2_leaf_reg=3.0,
        boosting_type='Plain', bootstrap_type='Bernoulli', subsample=0.8,
        eval_metric='AUC', loss_function='Logloss', cat_features=cat_idx,
        early_stopping_rounds=80, verbose=100, random_seed=42, thread_count=-1,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    oof_cat[va_idx] = model.predict_proba(X_va)[:, 1]
    test_cat += model.predict_proba(test[feat_cols])[:, 1] / n_splits
    print(f'CatBoost Fold {fold}: AUC={roc_auc_score(y_va, oof_cat[va_idx]):.5f} time={time.time()-t0:.0f}s', flush=True)
    np.save('models/stage2_cat_oof.npy', oof_cat)
    np.save('models/stage2_cat_test.npy', test_cat)

print(f'\nCatBoost OOF AUC: {roc_auc_score(target, oof_cat):.5f}', flush=True)
