import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import sys, time
sys.path.insert(0, 'src')
from nested_features import build_fold_features, get_outer_folds, CAT_COLS

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
target = (train['Will_Buy_EV'] == 'Yes').astype(int)
train = train.drop(columns=['Will_Buy_EV'])

folds = get_outer_folds(train, target, seed=42, cache_path='models/folds.npy')
n_splits = folds.max() + 1

LGB_PARAMS = dict(
    objective='binary', metric='auc', learning_rate=0.03,
    num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1, seed=42, max_bin=1024,
)
SHALLOW_PARAMS = dict(
    objective='binary', metric='auc', learning_rate=0.02,
    max_depth=5, num_leaves=31, min_child_samples=100, subsample=0.7, subsample_freq=1,
    colsample_bytree=0.3, reg_lambda=2.0, verbosity=-1, seed=777,
)
XGB_PARAMS = dict(
    objective='binary:logistic', eval_metric='auc', learning_rate=0.03,
    max_depth=7, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, tree_method='hist', enable_categorical=True,
)

results = {}
oof_lgb = np.zeros(len(train)); test_lgb = np.zeros(len(test))
oof_shallow = np.zeros(len(train)); test_shallow = np.zeros(len(test))
oof_xgb = np.zeros(len(train)); test_xgb = np.zeros(len(test))

for k in range(n_splits):
    t0 = time.time()
    outer_train_idx = np.where(folds != k)[0]
    outer_valid_idx = np.where(folds == k)[0]
    tr, va, te = build_fold_features(train, test, target, outer_train_idx, outer_valid_idx, seed=42)
    feat_cols = [c for c in tr.columns if c != 'id']
    y_tr = target.iloc[outer_train_idx].reset_index(drop=True)
    y_va = target.iloc[outer_valid_idx].reset_index(drop=True)

    for c in CAT_COLS:
        tr[c] = tr[c].astype('category')
        va[c] = va[c].astype('category')
        te[c] = te[c].astype('category')

    # main LGB
    dtr = lgb.Dataset(tr[feat_cols], y_tr, categorical_feature=CAT_COLS)
    dva = lgb.Dataset(va[feat_cols], y_va, categorical_feature=CAT_COLS, reference=dtr)
    m = lgb.train(LGB_PARAMS, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[outer_valid_idx] = m.predict(va[feat_cols], num_iteration=m.best_iteration)
    test_lgb += m.predict(te[feat_cols], num_iteration=m.best_iteration) / n_splits

    # shallow LGB
    dtr2 = lgb.Dataset(tr[feat_cols], y_tr, categorical_feature=CAT_COLS)
    dva2 = lgb.Dataset(va[feat_cols], y_va, categorical_feature=CAT_COLS, reference=dtr2)
    m2 = lgb.train(SHALLOW_PARAMS, dtr2, num_boost_round=4000, valid_sets=[dva2], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof_shallow[outer_valid_idx] = m2.predict(va[feat_cols], num_iteration=m2.best_iteration)
    test_shallow += m2.predict(te[feat_cols], num_iteration=m2.best_iteration) / n_splits

    # XGB
    dtr3 = xgb.DMatrix(tr[feat_cols], y_tr, enable_categorical=True)
    dva3 = xgb.DMatrix(va[feat_cols], y_va, enable_categorical=True)
    dte3 = xgb.DMatrix(te[feat_cols], enable_categorical=True)
    m3 = xgb.train(XGB_PARAMS, dtr3, num_boost_round=3000, evals=[(dva3, 'val')], early_stopping_rounds=100, verbose_eval=False)
    oof_xgb[outer_valid_idx] = m3.predict(dva3, iteration_range=(0, m3.best_iteration + 1))
    test_xgb += m3.predict(dte3, iteration_range=(0, m3.best_iteration + 1)) / n_splits

    print(f'fold {k} done in {time.time()-t0:.0f}s | lgb={roc_auc_score(y_va, oof_lgb[outer_valid_idx]):.5f} '
          f'shallow={roc_auc_score(y_va, oof_shallow[outer_valid_idx]):.5f} xgb={roc_auc_score(y_va, oof_xgb[outer_valid_idx]):.5f}', flush=True)

print('\n=== CLEAN NESTED-CV OOF (leak-free) ===')
print(f'lgb_clean:     {roc_auc_score(target, oof_lgb):.5f}   (old leaky: 0.94556)')
print(f'shallow_clean: {roc_auc_score(target, oof_shallow):.5f}   (old leaky: 0.94570)')
print(f'xgb_clean:     {roc_auc_score(target, oof_xgb):.5f}   (old leaky: 0.94554)')

np.save('models/clean_lgb_oof.npy', oof_lgb); np.save('models/clean_lgb_test.npy', test_lgb)
np.save('models/clean_shallow_oof.npy', oof_shallow); np.save('models/clean_shallow_test.npy', test_shallow)
np.save('models/clean_xgb_oof.npy', oof_xgb); np.save('models/clean_xgb_test.npy', test_xgb)
