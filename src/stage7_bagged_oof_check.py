"""P2: 2-repeat bagged-OOF sanity check for the strongest local model (shallow LGB, clean OOF 0.94560).
Question: does a second, independent 5-fold partition materially change this model's OOF or test predictions?
If repeat-to-repeat correlation is very high and AUCs are close, the single-partition OOF is trustworthy
and we stop here (per the stopping rule) rather than building a full bagging framework.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import sys, time
sys.path.insert(0, 'src')
from nested_features import build_fold_features, get_outer_folds, CAT_COLS

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
target = (train['Will_Buy_EV'] == 'Yes').astype(int)
train = train.drop(columns=['Will_Buy_EV'])

SHALLOW_PARAMS = dict(
    objective='binary', metric='auc', learning_rate=0.02,
    max_depth=5, num_leaves=31, min_child_samples=100, subsample=0.7, subsample_freq=1,
    colsample_bytree=0.3, reg_lambda=2.0, verbosity=-1, seed=777,
)

# repeat 1 = existing clean folds (seed 42, already cached)
# repeat 2 = an independent partition (seed 123)
folds_r1 = get_outer_folds(train, target, seed=42, cache_path='models/folds.npy')
folds_r2 = get_outer_folds(train, target, seed=123, cache_path='models/folds_repeat2.npy')
oof_r1 = np.load('models/clean_shallow_oof.npy')
test_r1 = np.load('models/clean_shallow_test.npy')

n_splits = folds_r2.max() + 1
oof_r2 = np.zeros(len(train))
test_r2 = np.zeros(len(test))

for k in range(n_splits):
    t0 = time.time()
    outer_train_idx = np.where(folds_r2 != k)[0]
    outer_valid_idx = np.where(folds_r2 == k)[0]
    tr, va, te = build_fold_features(train, test, target, outer_train_idx, outer_valid_idx, seed=123)
    feat_cols = [c for c in tr.columns if c != 'id']
    y_tr = target.iloc[outer_train_idx].reset_index(drop=True)
    y_va = target.iloc[outer_valid_idx].reset_index(drop=True)
    for c in CAT_COLS:
        tr[c] = tr[c].astype('category'); va[c] = va[c].astype('category'); te[c] = te[c].astype('category')
    dtr = lgb.Dataset(tr[feat_cols], y_tr, categorical_feature=CAT_COLS)
    dva = lgb.Dataset(va[feat_cols], y_va, categorical_feature=CAT_COLS, reference=dtr)
    m = lgb.train(SHALLOW_PARAMS, dtr, num_boost_round=4000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof_r2[outer_valid_idx] = m.predict(va[feat_cols], num_iteration=m.best_iteration)
    test_r2 += m.predict(te[feat_cols], num_iteration=m.best_iteration) / n_splits
    print(f'repeat2 fold {k} done in {time.time()-t0:.0f}s | auc={roc_auc_score(y_va, oof_r2[outer_valid_idx]):.5f}', flush=True)

oof_avg = (oof_r1 + oof_r2) / 2
test_avg = (test_r1 + test_r2) / 2

print('\n=== BAGGED-OOF SANITY CHECK (shallow LGB, 2 independent partitions) ===')
print(f'repeat1 OOF AUC: {roc_auc_score(target, oof_r1):.5f}')
print(f'repeat2 OOF AUC: {roc_auc_score(target, oof_r2):.5f}')
print(f'averaged OOF AUC: {roc_auc_score(target, oof_avg):.5f}')
print(f'repeat1 vs repeat2 OOF Spearman correlation: {spearmanr(oof_r1, oof_r2).correlation:.5f}')
print(f'repeat1 vs repeat2 TEST Spearman correlation: {spearmanr(test_r1, test_r2).correlation:.5f}')
print(f'repeat1 vs repeat2 test-pred mean abs diff: {np.mean(np.abs(test_r1 - test_r2)):.5f}')

np.save('models/clean_shallow_oof_repeat2.npy', oof_r2)
np.save('models/clean_shallow_test_repeat2.npy', test_r2)
np.save('models/clean_shallow_oof_bagged.npy', oof_avg)
np.save('models/clean_shallow_test_bagged.npy', test_avg)
