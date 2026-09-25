import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CAT_COLS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']

TE_SPECS = (
    [('Annual_Income_USD', bw, (20, 200)) for bw in (None, 500, 2000, 5000)]
    + [('Daily_Commute_km', bw, (20, 200)) for bw in (None, 2, 5)]
    + [(col, None, (10, 50)) for col in CAT_COLS]
)
GEN_SCORE_TE_SPECS = [(2, (10, 50)), (3, (5, 30))]


def add_generator_score(df):
    subsidy = (df['Subsidy_Available'] == 'Yes').astype(int)
    med_anx = (df['Range_Anxiety_Level'] == 'Medium').astype(int)
    high_anx = (df['Range_Anxiety_Level'] == 'High').astype(int)
    df['gen_score'] = (1.2 * (df['Annual_Income_USD'] / 100000) + 0.6 * df['Environmental_Concern_Level']
                        + 2 * subsidy - 1 * med_anx - 3 * high_anx)
    df['gen_score_thresh'] = (df['gen_score'] >= 5.5).astype(int)
    return df


def add_interactions(df):
    subsidy = (df['Subsidy_Available'] == 'Yes').astype(int)
    home_charge = (df['Home_Charging_Possible'] == 'Yes').astype(int)
    income_100k = df['Annual_Income_USD'] / 100000
    df['inter_income_subsidy'] = income_100k * subsidy
    df['inter_income_concern'] = income_100k * df['Environmental_Concern_Level']
    df['inter_concern_subsidy'] = df['Environmental_Concern_Level'] * subsidy
    df['inter_home_subsidy'] = home_charge * subsidy
    df['total_charging_stations'] = df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    df['charging_per_car'] = df['Charging_Stations_Near_Home'] / (df['Number_of_Cars_Owned'] + 1)
    df['commute_per_income'] = df['Daily_Commute_km'] / (df['Annual_Income_USD'] + 1)
    return df


def make_key(df, col, bin_width):
    if bin_width:
        return (df[col] // bin_width).astype(int)
    return df[col]


def fit_te_stats(key_series, target_series, smoothings, global_mean):
    stats = target_series.groupby(key_series.values).agg(['mean', 'count'])
    return {s: (stats['mean'] * stats['count'] + global_mean * s) / (stats['count'] + s) for s in smoothings}


def apply_te_stats(key_series, stats_dict, global_mean):
    return {s: key_series.map(m).fillna(global_mean).values for s, m in stats_dict.items()}


def inner_oof_te(key_full, target_full, smoothings, global_mean, n_inner=5, seed=42):
    """key_full/target_full are already positionally aligned (0..n-1) subsets of outer_train.
    Returns dict smoothing -> OOF-safe encoded array of the same length, computed with an
    independent inner K-fold that never touches outer_valid or test."""
    n = len(key_full)
    out = {s: np.zeros(n) for s in smoothings}
    inner_skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for itr, iva in inner_skf.split(key_full, target_full):
        stats = fit_te_stats(key_full.iloc[itr], target_full.iloc[itr], smoothings, global_mean)
        mapped = apply_te_stats(key_full.iloc[iva], stats, global_mean)
        for s in smoothings:
            out[s][iva] = mapped[s]
    return out


def get_outer_folds(train, target, n_splits=5, seed=42, cache_path=None):
    if cache_path:
        try:
            return np.load(cache_path)
        except FileNotFoundError:
            pass
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(train), -1, dtype=int)
    for k, (_, va_idx) in enumerate(skf.split(train, target)):
        folds[va_idx] = k
    if cache_path:
        np.save(cache_path, folds)
    return folds


def build_fold_features(train_raw, test_raw, target, outer_train_idx, outer_valid_idx, seed=42):
    """Zero-leakage feature builder for one outer fold.
    outer_train features: inner-cross-fit TE (never sees outer_valid or test labels).
    outer_valid / test features: TE fit on the FULL outer_train only.
    """
    tr = train_raw.iloc[outer_train_idx].reset_index(drop=True).copy()
    va = train_raw.iloc[outer_valid_idx].reset_index(drop=True).copy()
    te = test_raw.reset_index(drop=True).copy()
    y_tr = target.iloc[outer_train_idx].reset_index(drop=True)

    for df in (tr, va, te):
        add_generator_score(df)
        add_interactions(df)

    for col in ['Annual_Income_USD', 'Daily_Commute_km']:
        vc = tr[col].value_counts()
        tr[f'{col}_freq'] = tr[col].map(vc).fillna(0)
        va[f'{col}_freq'] = va[col].map(vc).fillna(0)
        te[f'{col}_freq'] = te[col].map(vc).fillna(0)

    global_mean = y_tr.mean()

    def process_te(col, bin_width, smoothings):
        key_name = f'{col}_bin{bin_width}' if bin_width else col
        key_tr = make_key(tr, col, bin_width)
        inner_vals = inner_oof_te(key_tr, y_tr, smoothings, global_mean, seed=seed)
        stats_full = fit_te_stats(key_tr, y_tr, smoothings, global_mean)
        key_va = make_key(va, col, bin_width)
        key_te = make_key(te, col, bin_width)
        mapped_va = apply_te_stats(key_va, stats_full, global_mean)
        mapped_te = apply_te_stats(key_te, stats_full, global_mean)
        for s in smoothings:
            tr[f'{key_name}_te_s{s}'] = inner_vals[s]
            va[f'{key_name}_te_s{s}'] = mapped_va[s]
            te[f'{key_name}_te_s{s}'] = mapped_te[s]

    for col, bw, smooths in TE_SPECS:
        process_te(col, bw, smooths)

    for prec, smooths in GEN_SCORE_TE_SPECS:
        key = f'gen_score_r{prec}'
        tr[key] = tr['gen_score'].round(prec)
        va[key] = va['gen_score'].round(prec)
        te[key] = te['gen_score'].round(prec)
        process_te(key, None, smooths)

    return tr, va, te
