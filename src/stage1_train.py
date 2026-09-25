import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, 'src')
from features import build_features, CAT_COLS

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
target = (train['Will_Buy_EV'] == 'Yes').astype(int)
train = train.drop(columns=['Will_Buy_EV'])

train, test = build_features(train, test, target)

for c in CAT_COLS:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')

feat_cols = [c for c in train.columns if c != 'id']
print(f'Total features: {len(feat_cols)}')

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof = np.zeros(len(train))
test_preds = np.zeros(len(test))

params = dict(
    objective='binary', metric='auc', learning_rate=0.03,
    num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1, seed=42, max_bin=1024,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train[feat_cols], target)):
    X_tr, y_tr = train.iloc[tr_idx][feat_cols], target.iloc[tr_idx]
    X_va, y_va = train.iloc[va_idx][feat_cols], target.iloc[va_idx]
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=CAT_COLS, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += model.predict(test[feat_cols], num_iteration=model.best_iteration) / n_splits
    fold_auc = roc_auc_score(y_va, oof[va_idx])
    print(f'Fold {fold}: AUC={fold_auc:.5f} best_iter={model.best_iteration}')

overall_auc = roc_auc_score(target, oof)
print(f'\nOOF AUC (stage1: +gen_score +freq +nested TE income/commute +cat TE): {overall_auc:.5f}')
print('(stage0 baseline was 0.94182; verified postmortem stages ranged 0.9436-0.9462)')

# feature importance
imp = pd.Series(model.feature_importance(importance_type='gain'), index=feat_cols).sort_values(ascending=False)
print('\nTop 20 features by gain:')
print(imp.head(20))

np.save('models/stage1_oof.npy', oof)
np.save('models/stage1_test.npy', test_preds)
train.to_parquet('data/train_features.parquet')
test.to_parquet('data/test_features.parquet')
