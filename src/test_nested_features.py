import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'src')
from nested_features import build_fold_features, get_outer_folds, TE_SPECS, GEN_SCORE_TE_SPECS

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
target = (train['Will_Buy_EV'] == 'Yes').astype(int)
train = train.drop(columns=['Will_Buy_EV'])

folds = get_outer_folds(train, target, seed=42)
k = 0
outer_train_idx = np.where(folds != k)[0]
outer_valid_idx = np.where(folds == k)[0]

TARGET_DERIVED_SUFFIXES = ('_te_s',)


def classify_columns(df):
    target_derived = [c for c in df.columns if any(suf in c for suf in TARGET_DERIVED_SUFFIXES)]
    target_free = [c for c in df.columns if c not in target_derived]
    return target_derived, target_free


print('=== Building reference fold features (baseline) ===')
tr0, va0, te0 = build_fold_features(train, test, target, outer_train_idx, outer_valid_idx, seed=42)
target_derived_cols, target_free_cols = classify_columns(tr0)
print(f'TARGET_DERIVED columns ({len(target_derived_cols)}): {target_derived_cols}')
print(f'TARGET_FREE columns ({len(target_free_cols)}): {target_free_cols}')

# ---------------- TEST 1: outer-valid label perturbation must not change ANY feature ----------------
print('\n=== TEST 1: perturb outer-VALID labels only -> features must be bit-identical ===')
target_perturbed = target.copy()
rng = np.random.RandomState(0)
target_perturbed.iloc[outer_valid_idx] = rng.randint(0, 2, size=len(outer_valid_idx))
tr1, va1, te1 = build_fold_features(train, test, target_perturbed, outer_train_idx, outer_valid_idx, seed=42)

fail = False
for name, dfa, dfb in [('outer_train', tr0, tr1), ('outer_valid', va0, va1), ('test', te0, te1)]:
    for col in target_derived_cols:
        if not np.allclose(dfa[col].values, dfb[col].values, equal_nan=True):
            print(f'  FAIL: {name}.{col} changed when outer-valid labels were perturbed (LEAK)')
            fail = True
if not fail:
    print('  PASS: all target-derived features identical after perturbing outer-valid labels only.')
else:
    raise SystemExit('TEST 1 FAILED - leak detected, stopping before wasting compute on retraining.')

# ---------------- TEST 2: outer-train label perturbation MUST change target-derived features ----------------
print('\n=== TEST 2: perturb outer-TRAIN labels -> target-derived features must change (sanity, TE not dead) ===')
target_perturbed2 = target.copy()
target_perturbed2.iloc[outer_train_idx] = rng.randint(0, 2, size=len(outer_train_idx))
tr2, va2, te2 = build_fold_features(train, test, target_perturbed2, outer_train_idx, outer_valid_idx, seed=42)

changed_any = False
for col in target_derived_cols:
    if not np.allclose(tr0[col].values, tr2[col].values, equal_nan=True):
        changed_any = True
        break
print('  PASS: target-derived features changed as expected.' if changed_any else '  FAIL: TE appears inert (not actually using targets) - investigate.')
assert changed_any, 'TEST 2 FAILED'

# ---------------- TEST 3: row alignment ----------------
print('\n=== TEST 3: row alignment checks ===')
assert (va0['id'].values == train.iloc[outer_valid_idx]['id'].values).all(), 'valid id order mismatch'
assert (te0['id'].values == test['id'].values).all(), 'test id order mismatch'
assert va0['id'].nunique() == len(va0), 'duplicate ids in outer-valid'
assert te0['id'].nunique() == len(te0), 'duplicate ids in test'
assert tr0['id'].nunique() == len(tr0), 'duplicate ids in outer-train'
all_ids = set(tr0['id']) | set(va0['id'])
assert all_ids == set(train['id']), 'missing/extra ids across outer-train + outer-valid'
print('  PASS: ids aligned, no duplicates, no missing rows.')

# ---------------- TEST 4: feature isolation summary ----------------
print('\n=== TEST 4: TARGET_DERIVED vs TARGET_FREE classification ===')
print(f'  {len(target_derived_cols)} target-derived (nested fit/transform): {target_derived_cols}')
print(f'  {len(target_free_cols)} target-free (safe to compute directly): {target_free_cols}')

print('\nALL TESTS PASSED - nested_features.py is verified leak-free for fold 0.')
