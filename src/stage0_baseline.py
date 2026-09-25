import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

target = (train['Will_Buy_EV'] == 'Yes').astype(int)
train = train.drop(columns=['Will_Buy_EV'])

cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
feat_cols = [c for c in train.columns if c != 'id']

for c in cat_cols:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof = np.zeros(len(train))
test_preds = np.zeros(len(test))

params = dict(
    objective='binary', metric='auc', learning_rate=0.03,
    num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1, seed=42,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train[feat_cols], target)):
    X_tr, y_tr = train.iloc[tr_idx][feat_cols], target.iloc[tr_idx]
    X_va, y_va = train.iloc[va_idx][feat_cols], target.iloc[va_idx]
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += model.predict(test[feat_cols], num_iteration=model.best_iteration) / n_splits
    fold_auc = roc_auc_score(y_va, oof[va_idx])
    print(f'Fold {fold}: AUC={fold_auc:.5f} best_iter={model.best_iteration}')

overall_auc = roc_auc_score(target, oof)
print(f'\nOOF AUC (raw 13 features, LightGBM): {overall_auc:.5f}')
print('(benchmark from verified postmortem: ~0.9419 for raw-feature LightGBM baseline)')

np.save('models/stage0_oof.npy', oof)
np.save('models/stage0_test.npy', test_preds)
