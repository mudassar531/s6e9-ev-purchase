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

# ---------------- TEST-ONLY / UNVALIDATED PUBLIC MEMBERS (no OOF -> never in the primary ensemble) ----------------
# EXCLUDED: data/public_kernels/amanatar_s6e9-samart-sota-meta-blend-lb-0-94656-champion/submission.csv
#   Its kernel log showed "Applying Exact Public Split Calibration Swaps" plus hardcoded
#   per-row score offsets ("Income Dead Zone", "Commute Extreme" etc.) tuned to the public
#   test split specifically -> leaderboard probing, not real signal. Won't generalize to the
#   private leaderboard. Local copy of its output was deleted; this note documents why it's
#   gone rather than silently vanishing. Never re-add it to a serious ensemble.
TEST_ONLY_UNVALIDATED_MODELS = [
    'data/public_kernels/vinay24baghira_s6e9-rank-averaging-ensemble-0-946-lb/submission.csv',
    'data/public_kernels/lamhuy8904_s6e9-94-6-transformer-and-gbdt-ensemble/submission.csv',
    'data/public_kernels/najiama_s6e9-electric-vehicle-oof-cv-0-94618-lb-0-94633/submission.csv',
    'data/public_kernels/denpugovkin_fifteen-teachers-fewer-voices-s6e9/submission.csv',
    'data/public_blends/submission_10.csv',
    'data/public_blends/submission_v3.csv',
]
extra_ranks = []
extra_names = []
for f in TEST_ONLY_UNVALIDATED_MODELS:
    try:
        df = pd.read_csv(f).set_index('id').reindex(test_id_order)
        col = df.columns[0]
        extra_ranks.append(rankdata(df[col].values) / len(df))
        extra_names.append(f)
    except Exception as e:
        print(f'skip {f}: {e}')

# ---- PRIMARY submission: OOF-validated models ONLY. This is the private-LB candidate. ----
sub_primary = pd.DataFrame({'id': test_id_order, 'Will_Buy_EV': core_test_blend})
sub_primary.to_csv('submissions/stage4_primary_oof_validated.csv', index=False)
print(f'\nSaved submissions/stage4_primary_oof_validated.csv (OOF={roc_auc_score(target, blend_oof):.5f}) - PRIMARY candidate, OOF-validated members only')

# ---- Separate, explicitly-labeled HIGH-RISK submission mixing in unvalidated public predictions ----
if extra_ranks:
    extra_mat = np.vstack(extra_ranks).T
    extra_avg = extra_mat.mean(axis=1)
    HIGH_RISK_WEIGHT = 0.25
    high_risk_score = (1 - HIGH_RISK_WEIGHT) * (rankdata(core_test_blend) / len(core_test_blend)) + HIGH_RISK_WEIGHT * extra_avg
    sub_risk = pd.DataFrame({'id': test_id_order, 'Will_Buy_EV': high_risk_score})
    sub_risk.to_csv('submissions/high_risk/stage4_with_unvalidated_pool.csv', index=False)
    print(f'Saved submissions/high_risk/stage4_with_unvalidated_pool.csv - NOT the primary candidate, no OOF backing for {len(extra_names)} pooled members: {extra_names}')
print(sub['Will_Buy_EV'].describe())
