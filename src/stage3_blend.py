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

# BUGFIX (previously): this block always submitted the linear-probability blend
# regardless of which strategy (linear vs rank) actually won on OOF. The historic
# 0.94638 submission was therefore a probability blend, not a rank blend as labeled.
# See reports/historic_stage3_weights.json for the reconstructed record of that run.
# Now: compute and save BOTH correctly, using the right test-side transform for each.
test_mat = np.vstack(tests).T
test_rank_mat = np.apply_along_axis(rankdata, 0, test_mat) / test_mat.shape[0]

prob_auc = roc_auc_score(target, blend_oof)
rank_auc = roc_auc_score(target, blend_oof_rank)
print(f'\nprobability blend OOF={prob_auc:.5f}  |  rank blend OOF={rank_auc:.5f}')

np.save('models/blend_oof.npy', blend_oof)
np.save('models/blend_weights.npy', w)
np.save('models/blend_rank_weights.npy', wr)

sub_prob = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': test_mat @ w})
sub_prob.to_csv('submissions/stage3_probability_blend.csv', index=False)
print('Saved submissions/stage3_probability_blend.csv')

sub_rank = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': test_rank_mat @ wr})
sub_rank.to_csv('submissions/stage3_rank_blend.csv', index=False)
print('Saved submissions/stage3_rank_blend.csv')
