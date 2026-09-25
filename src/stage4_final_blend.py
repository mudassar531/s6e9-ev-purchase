import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import rankdata

orig = pd.read_csv('data/train.csv')
target = (orig['Will_Buy_EV'] == 'Yes').astype(int).values
test = pd.read_csv('data/test.csv')
id_order = orig['id'].values
test_id_order = test['id'].values

# ---------------- OOF-VALIDATED MEMBERS (real weight optimization) ----------------
oof_members = {}   # name -> (oof_array_aligned_to_id_order, test_array_aligned_to_test_id_order)

def add_npy(name, oof_path, test_path):
    oof_members[name] = (np.load(oof_path), np.load(test_path))

def add_csv(name, oof_path, oof_col, test_path, test_col=None):
    odf = pd.read_csv(oof_path).set_index('id').reindex(id_order)
    tdf = pd.read_csv(test_path).set_index('id').reindex(test_id_order)
    tc = test_col if (test_col and test_col in tdf.columns) else tdf.columns[0]
    oof_members[name] = (odf[oof_col].values, tdf[tc].values)

add_npy('lgb', 'models/stage1_oof.npy', 'models/stage1_test.npy')
add_npy('xgb', 'models/stage2_xgb_oof.npy', 'models/stage2_xgb_test.npy')
add_npy('lgb_shallow', 'models/stage2c_shallow_oof.npy', 'models/stage2c_shallow_test.npy')

add_csv('pub_najiama_xgb_te', 'data/public_kernels/najiama_xgboost-triple-te-dynamic-pruning-lb-0-94639/oof_XGBOOST.csv', 'OOF_Pred',
        'data/public_kernels/najiama_xgboost-triple-te-dynamic-pruning-lb-0-94639/test_XGBOOST.csv')
add_csv('pub_aryankaisth_hist', 'data/public_kernels/aryankaisth_signal-that-matters-eda-histgbm-lb-0-94637/oof_proba.csv', 'Will_Buy_EV',
        'data/public_kernels/aryankaisth_signal-that-matters-eda-histgbm-lb-0-94637/test_proba.csv')
add_csv('pub_najiama_lgbm', 'data/public_kernels/najiama_pure-lgbm-model-cv-0-94607-lb-0-94638/oof_LIGHTGBM.csv', 'OOF_Pred',
        'data/public_kernels/najiama_pure-lgbm-model-cv-0-94607-lb-0-94638/test_LIGHTGBM.csv')
add_npy('pub_evgendvorkin_xgb', 'data/public_kernels/evgendvorkin_s6e9-single-xgb-cv-0-94583/oof_preds_base.npy',
        'data/public_kernels/evgendvorkin_s6e9-single-xgb-cv-0-94583/test_preds_base.npy')
add_csv('pub_yekenot_realmlp', 'data/public_kernels/yekenot_ps-s6-e9-realmlp-pytorch/oof_preds.csv', 'Will_Buy_EV',
        'data/public_kernels/yekenot_ps-s6-e9-realmlp-pytorch/submission.csv')

names = list(oof_members.keys())
oof_mat = np.vstack([oof_members[n][0] for n in names]).T
test_mat_raw = np.vstack([oof_members[n][1] for n in names]).T

print('OOF-validated members:')
for n in names:
    print(f'  {n}: AUC={roc_auc_score(target, oof_members[n][0]):.5f}')

ranks = np.apply_along_axis(rankdata, 0, oof_mat)
rank_mat = ranks / ranks.shape[0]

def neg_auc_rank(w, mat, y):
    w = np.abs(w); w = w / w.sum()
    return -roc_auc_score(y, mat @ w)

x0 = np.ones(len(names)) / len(names)
res = minimize(neg_auc_rank, x0, args=(rank_mat, target), method='Nelder-Mead',
                options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 8000})
w = np.abs(res.x); w = w / w.sum()
print('\nOptimized rank weights:', dict(zip(names, w.round(4))))
blend_oof = rank_mat @ w
print('Optimized rank blend OOF AUC:', roc_auc_score(target, blend_oof))
print('Equal-weight rank blend OOF AUC:', roc_auc_score(target, rank_mat.mean(axis=1)))

# rank the test matrix using the SAME procedure (independent ranks within test set)
test_ranks = np.apply_along_axis(rankdata, 0, test_mat_raw) / test_mat_raw.shape[0]
core_test_blend = test_ranks @ w

# ---------------- TEST-ONLY PUBLIC MEMBERS (unvalidated -> small fixed weight pool) ----------------
extra_test_files = [
    'data/public_kernels/amanatar_s6e9-samart-sota-meta-blend-lb-0-94656-champion/submission.csv',
    'data/public_kernels/vinay24baghira_s6e9-rank-averaging-ensemble-0-946-lb/submission.csv',
    'data/public_kernels/lamhuy8904_s6e9-94-6-transformer-and-gbdt-ensemble/submission.csv',
    'data/public_kernels/najiama_s6e9-electric-vehicle-oof-cv-0-94618-lb-0-94633/submission.csv',
    'data/public_kernels/denpugovkin_fifteen-teachers-fewer-voices-s6e9/submission.csv',
    'data/public_blends/submission_10.csv',
    'data/public_blends/submission_v3.csv',
]
extra_ranks = []
extra_names = []
for f in extra_test_files:
    try:
        df = pd.read_csv(f).set_index('id').reindex(test_id_order)
        col = df.columns[0]
        extra_ranks.append(rankdata(df[col].values) / len(df))
        extra_names.append(f)
    except Exception as e:
        print(f'skip {f}: {e}')

extra_mat = np.vstack(extra_ranks).T if extra_ranks else None
print(f'\n{len(extra_names)} extra unvalidated test-only members included at low weight')

# final: core (OOF-validated, high trust) gets 75%, unvalidated public pool gets 25% (equal-avg within pool)
import os
CORE_WEIGHT = float(os.environ.get('CORE_WEIGHT', 0.75))
if extra_mat is not None:
    extra_avg = extra_mat.mean(axis=1)
    final_test_score = CORE_WEIGHT * rankdata(core_test_blend) / len(core_test_blend) + (1 - CORE_WEIGHT) * extra_avg
else:
    final_test_score = core_test_blend

sub = pd.DataFrame({'id': test_id_order, 'Will_Buy_EV': final_test_score})
out_path = f'submissions/stage4_final_blend_core{CORE_WEIGHT}.csv'
sub.to_csv(out_path, index=False)
print(f'\nSaved {out_path}')
print(sub['Will_Buy_EV'].describe())
