import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, 'src')
from features import CAT_COLS

train = pd.read_parquet('data/train_features.parquet')
test = pd.read_parquet('data/test_features.parquet')
orig = pd.read_csv('data/train.csv')
target = (orig['Will_Buy_EV'] == 'Yes').astype(int)

for c in CAT_COLS:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')

feat_cols = [c for c in train.columns if c != 'id']
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)  # different seed for diversity

oof = np.zeros(len(train))
test_preds = np.zeros(len(test))

params = dict(
    objective='binary', metric='auc', learning_rate=0.02,
    max_depth=5, num_leaves=31, min_child_samples=100, subsample=0.7, subsample_freq=1,
    colsample_bytree=0.3, reg_lambda=2.0, verbosity=-1, seed=777,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train[feat_cols], target)):
    X_tr, y_tr = train.iloc[tr_idx][feat_cols], target.iloc[tr_idx]
    X_va, y_va = train.iloc[va_idx][feat_cols], target.iloc[va_idx]
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=CAT_COLS, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += model.predict(test[feat_cols], num_iteration=model.best_iteration) / n_splits
    print(f'Fold {fold}: AUC={roc_auc_score(y_va, oof[va_idx]):.5f}', flush=True)

print(f'\nShallow LGB OOF AUC: {roc_auc_score(target, oof):.5f}')
np.save('models/stage2c_shallow_oof.npy', oof)
np.save('models/stage2c_shallow_test.npy', test_preds)
