import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

CAT_COLS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']


def add_generator_score(df):
    subsidy = (df['Subsidy_Available'] == 'Yes').astype(int)
    med_anx = (df['Range_Anxiety_Level'] == 'Medium').astype(int)
    high_anx = (df['Range_Anxiety_Level'] == 'High').astype(int)
    df['gen_score'] = (1.2 * (df['Annual_Income_USD'] / 100000) + 0.6 * df['Environmental_Concern_Level']
                        + 2 * subsidy - 1 * med_anx - 3 * high_anx)
    df['gen_score_thresh'] = (df['gen_score'] >= 5.5).astype(int)
    return df


def add_freq_encoding(train, test, cols):
    for col in cols:
        vc = train[col].value_counts()
        train[f'{col}_freq'] = train[col].map(vc).fillna(0)
        test[f'{col}_freq'] = test[col].map(vc).fillna(0)
    return train, test


def kfold_target_encode(train, test, col, target, n_splits=5, smoothings=(10, 100), seed=42, bin_width=None):
    """OOF-safe target encoding. If bin_width given, bins the numeric col first."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    key = f'{col}_bin{bin_width}' if bin_width else col
    if bin_width:
        train[key] = (train[col] // bin_width).astype(int)
        test[key] = (test[col] // bin_width).astype(int)

    global_mean = target.mean()
    new_cols_train = {f'{key}_te_s{s}': np.zeros(len(train)) for s in smoothings}

    for tr_idx, va_idx in skf.split(train, target):
        tr_key = train.iloc[tr_idx][key]
        tr_target = target.iloc[tr_idx]
        stats = tr_target.groupby(tr_key).agg(['mean', 'count'])
        for s in smoothings:
            smooth = (stats['mean'] * stats['count'] + global_mean * s) / (stats['count'] + s)
            mapped = train.iloc[va_idx][key].map(smooth).fillna(global_mean)
            new_cols_train[f'{key}_te_s{s}'][va_idx] = mapped.values

    for name, arr in new_cols_train.items():
        train[name] = arr

    # test uses full-train stats
    stats_full = target.groupby(train[key]).agg(['mean', 'count'])
    for s in smoothings:
        smooth = (stats_full['mean'] * stats_full['count'] + global_mean * s) / (stats_full['count'] + s)
        test[f'{key}_te_s{s}'] = test[key].map(smooth).fillna(global_mean)

    if bin_width:
        train.drop(columns=[key], inplace=True)
        test.drop(columns=[key], inplace=True)

    return train, test


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


def build_features(train, test, target):
    train = add_generator_score(train)
    test = add_generator_score(test)
    train = add_interactions(train)
    test = add_interactions(test)

    train, test = add_freq_encoding(train, test, ['Annual_Income_USD', 'Daily_Commute_km'])

    # nested target encoding at multiple bin widths for income and commute
    for bw in [1, 500, 2000, 5000]:
        train, test = kfold_target_encode(train, test, 'Annual_Income_USD', target, bin_width=(None if bw == 1 else bw), smoothings=(20, 200))
    for bw in [1, 2, 5]:
        train, test = kfold_target_encode(train, test, 'Daily_Commute_km', target, bin_width=(None if bw == 1 else bw), smoothings=(20, 200))

    # categorical target encoding at two smoothings
    for col in CAT_COLS:
        train, test = kfold_target_encode(train, test, col, target, smoothings=(10, 50))

    # gen_score binned target encoding (fine resolution on the derived score itself)
    for prec, smooths in [(2, (10, 50)), (3, (5, 30))]:
        key = f'gen_score_r{prec}'
        train[key] = train['gen_score'].round(prec)
        test[key] = test['gen_score'].round(prec)
        train, test = kfold_target_encode(train, test, key, target, smoothings=smooths)

    return train, test
