import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import rankdata

orig = pd.read_csv('data/train.csv')
target = (orig['Will_Buy_EV'] == 'Yes').astype(int).values
test = pd.read_csv('data/test.csv')

members = {
    'lgb': ('models/stage1_oof.npy', 'models/stage1_test.npy'),
    'xgb': ('models/stage2_xgb_oof.npy', 'models/stage2_xgb_test.npy'),
    'cat': ('models/stage2_cat_oof.npy', 'models/stage2_cat_test.npy'),
}

oofs, tests, names = [], [], []
for name, (oof_p, test_p) in members.items():
    try:
        oofs.append(np.load(oof_p))
        tests.append(np.load(test_p))
        names.append(name)
    except FileNotFoundError:
        print(f'skip {name}, not found')

# public notebook OOF members (verified against real train labels)
public_oof_members = {
    'pub_najiama_xgb': ('data/public_kernels/najiama_xgboost-triple-te-dynamic-pruning-lb-0-94639/oof_XGBOOST.csv', 'OOF_Pred',
                         'data/public_kernels/najiama_xgboost-triple-te-dynamic-pruning-lb-0-94639/test_XGBOOST.csv', 'pred'),
    'pub_aryankaisth_hist': ('data/public_kernels/aryankaisth_signal-that-matters-eda-histgbm-lb-0-94637/oof_proba.csv', 'Will_Buy_EV',
                              'data/public_kernels/aryankaisth_signal-that-matters-eda-histgbm-lb-0-94637/test_proba.csv', 'Will_Buy_EV'),
}
id_order = orig['id'].values
test_id_order = test['id'].values
for name, (oof_path, oof_col, test_path, test_col) in public_oof_members.items():
    try:
        odf = pd.read_csv(oof_path).set_index('id').reindex(id_order)
        tdf = pd.read_csv(test_path)
        tcol = test_col if test_col in tdf.columns else tdf.columns[1]
        tdf = tdf.set_index('id').reindex(test_id_order)
        oofs.append(odf[oof_col].values)
        tests.append(tdf[tcol].values)
        names.append(name)
    except Exception as e:
        print(f'skip {name}: {e}')

for n, o in zip(names, oofs):
    print(f'{n}: standalone OOF AUC = {roc_auc_score(target, o):.5f}')

# pairwise correlation
oof_mat = np.vstack(oofs).T
print('\nPairwise OOF rank-correlation:')
ranks = np.apply_along_axis(rankdata, 0, oof_mat)
corr = np.corrcoef(ranks.T)
print(pd.DataFrame(corr, index=names, columns=names).round(5))

# ---- Optimize linear blend weights on raw probability scale ----
def neg_auc(w, mat, y):
    w = np.abs(w)
    w = w / w.sum()
    blend = mat @ w
    return -roc_auc_score(y, blend)

x0 = np.ones(len(names)) / len(names)
res = minimize(neg_auc, x0, args=(oof_mat, target), method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 5000})
w = np.abs(res.x)
w = w / w.sum()
print('\nOptimized linear weights (prob scale):', dict(zip(names, w.round(4))))
blend_oof = oof_mat @ w
print('Linear blend OOF AUC:', roc_auc_score(target, blend_oof))

# ---- Rank blend ----
rank_mat = ranks / ranks.shape[0]
def neg_auc_rank(w, mat, y):
    w = np.abs(w)
    w = w / w.sum()
    return -roc_auc_score(y, mat @ w)
res_r = minimize(neg_auc_rank, x0, args=(rank_mat, target), method='Nelder-Mead',
                  options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 5000})
wr = np.abs(res_r.x)
wr = wr / wr.sum()
print('\nOptimized rank-blend weights:', dict(zip(names, wr.round(4))))
blend_oof_rank = rank_mat @ wr
print('Rank blend OOF AUC:', roc_auc_score(target, blend_oof_rank))

# simple equal-weight comparisons
print('\nEqual-weight prob blend AUC:', roc_auc_score(target, oof_mat.mean(axis=1)))
print('Equal-weight rank blend AUC:', roc_auc_score(target, rank_mat.mean(axis=1)))

# pick the best strategy automatically
test_mat = np.vstack(tests).T
candidates = {
    'linear_opt': (roc_auc_score(target, blend_oof), test_mat @ w),
    'rank_opt': (roc_auc_score(target, blend_oof_rank), None),  # need rank of test too
}
best_name = max(candidates, key=lambda k: candidates[k][0])
print(f'\nBest strategy: {best_name} with OOF {candidates[best_name][0]:.5f}')

np.save('models/blend_oof.npy', blend_oof)
np.save('models/blend_weights.npy', w)
final_test = test_mat @ w
sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': final_test})
sub.to_csv('submissions/stage3_blend.csv', index=False)
print('\nSaved submissions/stage3_blend.csv')
