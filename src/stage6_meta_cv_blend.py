import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize, differential_evolution
from scipy.stats import rankdata

orig = pd.read_csv('data/train.csv')
target = (orig['Will_Buy_EV'] == 'Yes').astype(int).values
id_order = orig['id'].values

oof_members = {}


def add_npy(name, path):
    oof_members[name] = np.load(path)


def add_csv(name, path, col):
    df = pd.read_csv(path).set_index('id').reindex(id_order)
    oof_members[name] = df[col].values


# clean (leak-fixed) local models
add_npy('lgb_clean', 'models/clean_lgb_oof.npy')
add_npy('shallow_clean', 'models/clean_shallow_oof.npy')
add_npy('xgb_clean', 'models/clean_xgb_oof.npy')
# verified public OOF models (independently produced, unaffected by our leak)
add_csv('pub_najiama_xgb_te', 'data/public_kernels/najiama_xgboost-triple-te-dynamic-pruning-lb-0-94639/oof_XGBOOST.csv', 'OOF_Pred')
add_csv('pub_aryankaisth_hist', 'data/public_kernels/aryankaisth_signal-that-matters-eda-histgbm-lb-0-94637/oof_proba.csv', 'Will_Buy_EV')
add_csv('pub_najiama_lgbm', 'data/public_kernels/najiama_pure-lgbm-model-cv-0-94607-lb-0-94638/oof_LIGHTGBM.csv', 'OOF_Pred')
add_npy('pub_evgendvorkin_xgb', 'data/public_kernels/evgendvorkin_s6e9-single-xgb-cv-0-94583/oof_preds_base.npy')
add_csv('pub_yekenot_realmlp', 'data/public_kernels/yekenot_ps-s6-e9-realmlp-pytorch/oof_preds.csv', 'Will_Buy_EV')

names = list(oof_members.keys())
oof_mat = np.vstack([oof_members[n] for n in names]).T
n, m = oof_mat.shape
print(f'{m} members, {n} rows')
for nm in names:
    print(f'  {nm}: standalone OOF AUC = {roc_auc_score(target, oof_members[nm]):.5f}')

rank_mat = np.apply_along_axis(rankdata, 0, oof_mat) / n

meta_folds = np.load('models/folds.npy')
n_meta = meta_folds.max() + 1


# ---------------- blend strategies ----------------
def greedy_forward(mat, y, max_rounds=25, tol=1e-6):
    n_models = mat.shape[1]
    picked = []
    cur = np.zeros(len(y))
    best_auc = 0.5
    for _ in range(max_rounds):
        best_gain, best_i = 0.0, None
        for i in range(n_models):
            k = len(picked) + 1
            trial = (cur * len(picked) + mat[:, i]) / k
            auc = roc_auc_score(y, trial)
            if auc - best_auc > best_gain:
                best_gain, best_i = auc - best_auc, i
        if best_i is None or best_gain < tol:
            break
        picked.append(best_i)
        cur = mat[:, picked].mean(axis=1)
        best_auc = roc_auc_score(y, cur)
    w = np.zeros(n_models)
    for i in picked:
        w[i] += 1
    return (w / w.sum()) if w.sum() > 0 else np.ones(n_models) / n_models


def dirichlet_search_refine(mat, y, n_samples=400, seed=0):
    rng = np.random.RandomState(seed)
    # vectorized: sample all weight vectors at once, score each with a single matmul batch
    ws = rng.dirichlet(np.ones(mat.shape[1]), size=n_samples)  # (n_samples, n_models)
    best_w, best_auc = ws[0], -1
    for w in ws:
        auc = roc_auc_score(y, mat @ w)
        if auc > best_auc:
            best_auc, best_w = auc, w

    def neg_auc(raw):
        w = np.abs(raw); w = w / w.sum()
        return -roc_auc_score(y, mat @ w)

    res = minimize(neg_auc, best_w, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-8, 'maxiter': 300})
    w = np.abs(res.x); w = w / w.sum()
    return w


def equal_weight(mat):
    return np.ones(mat.shape[1]) / mat.shape[1]


import os
STRATEGIES = {
    'equal': lambda mat, y: equal_weight(mat),
    'greedy': greedy_forward,
}
if os.environ.get('WITH_DIRICHLET'):
    STRATEGIES['dirichlet+refine'] = dirichlet_search_refine

for scale_name, mat in [('probability', oof_mat), ('rank', rank_mat)]:
    print(f'\n{"="*20} {scale_name.upper()} SCALE {"="*20}')
    best_single_idx = int(np.argmax([roc_auc_score(target, mat[:, i]) for i in range(m)]))
    print(f'best single model: {names[best_single_idx]}')

    for strat_name, fn in STRATEGIES.items():
        in_sample_aucs, held_out_aucs, weight_rows = [], [], []
        for k in range(n_meta):
            meta_tr = meta_folds != k
            meta_va = meta_folds == k
            w = fn(mat[meta_tr], target[meta_tr])
            in_sample_aucs.append(roc_auc_score(target[meta_tr], mat[meta_tr] @ w))
            held_out_aucs.append(roc_auc_score(target[meta_va], mat[meta_va] @ w))
            weight_rows.append(w)
        weight_rows = np.array(weight_rows)
        print(f'\n  [{strat_name}]')
        print(f'    in-sample OOF AUC (mean over folds): {np.mean(in_sample_aucs):.5f}')
        print(f'    HELD-OUT meta-CV AUC (mean):         {np.mean(held_out_aucs):.5f}  (std {np.std(held_out_aucs):.5f})')
        print(f'    weight stability (mean/std per model):')
        for i, nm in enumerate(names):
            print(f'      {nm:24s} mean={weight_rows[:,i].mean():.3f} std={weight_rows[:,i].std():.3f} '
                  f'min={weight_rows[:,i].min():.3f} max={weight_rows[:,i].max():.3f}')

    # full-data weights (for the eventual submission) using the winning strategy will be chosen after seeing this report
    full_greedy = greedy_forward(mat, target)
    print(f'\n  full-data greedy weights:  {dict(zip(names, full_greedy.round(3)))}')
    if os.environ.get('WITH_DIRICHLET'):
        full_dr = dirichlet_search_refine(mat, target)
        print(f'  full-data dirichlet weights: {dict(zip(names, full_dr.round(3)))}')
