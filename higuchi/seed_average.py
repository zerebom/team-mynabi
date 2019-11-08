# coding:utf-8
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from scipy.stats import norm, skew
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import datetime
import seaborn as sns
sns.set()
import matplotlib

pd.set_option('max_columns', 1000)
pd.set_option('max_rows', 1000)


import warnings
warnings.filterwarnings('ignore')

import re
import geocoder
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import os
import gc
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from time import time
import datetime
from script import RegressionPredictor, LogRegressionPredictor
import japanize_matplotlib
from utils import save_data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
SEED = 1234
n_splits = 10


def feature_encoding(train, test, category_col, target_col, func_list):
    '''target_encodingを重要な列（面積など）でやる。TEと違って、test_dfに含まれる値も集計して作る'''
    data = pd.concat([train, test], axis=0).reset_index()

    agg_func = {target_col: func_list}
    # agg_funcでgruopby
    agg_df = data.groupby(category_col)[target_col].agg(agg_func)
    # 列名作成
    agg_df.columns = [category_col + '_' + '_'.join(col).strip() for col in agg_df.columns.values]
    # 元の列に集約結果をmapしその値をコピーし新規列に加え返す。
    for col in agg_df.columns.values:
        train[col] = train[category_col].map(agg_df[col]).copy()
        test[col] = test[category_col].map(agg_df[col]).copy()
    return train, test


def target_encoding(train, test, category_col, target_col, func_list):
    '''target_encodingをやる。func_listに辞書型で列と処理する関数(meanとか)を渡す'''

    agg_func = {target_col: func_list}
    # agg_funcでgruopby
    agg_df = train.groupby(category_col)[target_col].agg(agg_func)
    # 列名作成
    agg_df.columns = [category_col + '_' + '_'.join(col).strip() for col in agg_df.columns.values]
    # 元の列に集約結果をmapしその値をコピーし新規列に加え返す。
    for col in agg_df.columns.values:
        train[col] = train[category_col].map(agg_df[col]).copy()
        test[col] = test[category_col].map(agg_df[col]).copy()
    return train, test


def get_Kmeans(X_train, X_test, n_clusters=300, cols=['loc_lat', 'loc_lon', '築年数'], out_col='km_type_with_age', seed=42):
    std = StandardScaler()

    X = X_train[cols]
    X2 = X_test[cols]
    Z = pd.concat([X, X2], axis=0)
    Z = std.fit_transform(Z)

    pred = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(Z)
    X_train[out_col] = pred[:X_train.shape[0]]
    X_test[out_col] = pred[X_train.shape[0]:]
    return X_train, X_test


train = pd.read_csv('../higuchi/input/prep_train1107.csv')
test = pd.read_csv('../higuchi/input/prep_test1107.csv')
drop_col = ['id']

# 必要な特徴量に絞る
y_train = train['賃料']
y_train_log = np.log1p(y_train)
y_train_area = np.log1p(train['賃料'] / train['面積'])

X_train = train.drop(drop_col, axis=1)
X_test = test.drop(drop_col, axis=1)
X_train['地価'] = X_train['賃料'] / X_train['面積']
X_train_raw = X_train.copy()
X_test_raw = X_test.copy()


for i in tqdm(range(10)):
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    X_train, X_test = get_Kmeans(X_train, X_test, n_clusters=300, seed=i)
    print('finish_kmeans')
    category_col = 'km_type_with_age'
    target_dict = {category_col: ['mean']}

    for category_col, func_list in target_dict.items():
        X_train, X_test = feature_encoding(X_train, X_test, category_col, '地価', func_list)
        X_train, X_test = feature_encoding(X_train, X_test, category_col, '面積', func_list)
        X_train, X_test = feature_encoding(X_train, X_test, category_col, '築年数', func_list)

    X_train['地価x面積'] = X_train['面積'] * X_train[f'{category_col}_地価_mean']
    X_test['地価x面積'] = X_test['面積'] * X_test[f'{category_col}_地価_mean']

    # 自身の面積と
    X_train['area_diff'] = X_train['面積'] - X_train[f'{category_col}_面積_mean']
    X_train['area_ratio'] = X_train['面積'] / X_train[f'{category_col}_面積_mean']
    X_test['area_diff'] = X_test['面積'] - X_test[f'{category_col}_面積_mean']
    X_test['area_ratio'] = X_test['面積'] / X_test[f'{category_col}_面積_mean']

    X_train['age_diff'] = X_train['築年数'] - X_train[f'{category_col}_築年数_mean']
    X_train['age_ratio'] = X_train['築年数'] / X_train[f'{category_col}_築年数_mean']
    X_test['age_diff'] = X_test['築年数'] - X_test[f'{category_col}_築年数_mean']
    X_test['age_ratio'] = X_test['築年数'] / X_test[f'{category_col}_築年数_mean']

    X_train['area/ageratio'] = X_train['area_ratio'] / X_train['age_ratio']
    X_test['area/ageratio'] = X_test['area_ratio'] / X_test['age_ratio']

    X_train['地価x面積'] = X_train['面積'] * X_train[f'{category_col}_地価_mean']
    X_test['地価x面積'] = X_test['面積'] * X_test[f'{category_col}_地価_mean']
    X_train['地価x面積/築年数ratio'] = X_train['地価x面積'] / X_train['age_ratio']
    X_test['地価x面積/築年数ratio'] = X_test['地価x面積'] / X_test['age_ratio']
    folder = KFold(n_splits=10, shuffle=True, random_state=i)
    lgbm_params = {
        'num_iterations': 50000,
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'reg_lambda': 0.016248748215948162,
        'reg_alpha': 0.017583014822381796,
        'colsample_bytree': 0.7,
        'subsample': 1.0,
        'max_depth': 8,
        'num_leaves':150,
        'min_child_weight': 25,
        'lambda': 1,
        'verbose': -1,
        'random_state': i,
        'early_stopping_round': 50
    }

    lgbm_log_params = lgbm_params.copy()
    lgbm_log_params['metric'] = 'mae'
    lgbm_log_params['learning_rate'] = 0.05

    features = ['面積', '築年数', 'center_dis', 'L', 'loc_lat', 'loc_lon', '総階数',
                '所在階', 'km_type_with_age_地価_mean', '地価x面積', 'area_diff',
                'area_ratio', 'age_diff', 'age_ratio', '地価x面積/築年数ratio', '地価_neighbor_1', '利用可能駅最大乗降人数', '上昇率']

    X_train_ = X_train.copy()
    X_test_ = X_test.copy()

    X_train = X_train[features]
    X_test = X_test[features]

    X_train.columns = X_train.columns.str.encode('utf-8').str.decode('utf-8')
    X_test.columns = X_test.columns.str.encode('utf-8').str.decode('utf-8')
    cols = ['area', 'age',  # 'sta_min',
            'center_dis', 'L', 'loc_lat', 'loc_lon', 'total_floor',
            # 'tatami',
            'floor', 'km_type_with_age_area_mean', 'land_pricexarea', 'area_diff',
            'area_ratio', 'age_diff',
            'age_ratio',  # 'area/ageratio',
            'land_pricexarea/ageratio',
            # 'lp_neighbor_mean', #'lp_neighbor_std', 'lp(area)_neighbor_mean', 'lp(area)_neighbor_std',
            'lp_neighbor_1',  # 'lp(area)_neighbor_1',
            #'avail_station', 'avail_root',
            #'sta_min_m', 'sta_min_people',
            'sta_max_people',
            'upper_rate',
            ]
    X_train.columns = cols
    X_test.columns = cols

    LogLGBM = LogRegressionPredictor(X_train, y_train.values, X_test, Folder=folder, params=lgbm_log_params,
                                     sk_model=None, n_splits=10, clf_type='lgb')
    log_lgboof, log_lgbpreds, log_lgbFIs = LogLGBM.fit()

    save_data(X_train, log_lgboof, log_lgbpreds, rmse=LogLGBM.rmse(), name=f'log_seed_{i}', save_dir='./seed_average')

    X_train_area = X_train_.copy()
    X_test_area = X_test_.copy()

    features = ['面積', '築年数',
                'center_dis',
                'loc_lat', 'loc_lon', '総階数', '畳', '所在階', 'km_type_with_age_地価_mean', '地価x面積', 'area_diff',
                'area_ratio', 'age_diff', 'age_ratio',
                'area/ageratio', '地価x面積/築年数ratio',
                '地価(単位面積)_neighbor_1',
                '最短駅m', '最短駅乗降人数', '利用可能駅最大乗降人数',
                '上昇率'
                ]
    X_train_area = X_train_area[features]
    X_test_area = X_test_area[features]

    X_train_area.columns = X_train_area.columns.str.encode('utf-8').str.decode('utf-8')
    X_test_area.columns = X_test_area.columns.str.encode('utf-8').str.decode('utf-8')

    cols = ['area', 'age',
            'center_dis',
            'loc_lat', 'loc_lon', 'total_floor',
            'tatami', 'floor', 'km_type_with_age_area_mean', 'land_pricexarea', 'area_diff',
            'area_ratio', 'age_diff', 'age_ratio', 'area/ageratio',
            'land_pricexarea/ageratio',
            'lp(area)_neighbor_1',
            'sta_min_m', 'sta_min_people', 'sta_max_people',
            'upper_rate'
            ]
    X_train_area.columns = cols
    X_test_area.columns = cols

    LogLGBM_area = LogRegressionPredictor(X_train_area, y_train_area.values, X_test_area, Folder=folder, params=lgbm_log_params,
                                          sk_model=None, n_splits=10, clf_type='lgb')
    log_lgboof_area, log_lgbpreds_area, log_lgbFIs_area = LogLGBM_area.fit()

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    area_oof = np.exp(log_lgboof_area) * X_train_area['area'].values
    area_pred = np.exp(log_lgbpreds_area) * X_test_area['area'].values

    normal_rmse = int(rmse(log_lgboof, y_train))
    area_rmse = int(rmse(area_oof, y_train))
    mean_rmse = int(rmse((log_lgboof + area_oof) / 2, y_train))

    print('賃料予測: ', normal_rmse)
    print('単位面積予測: ', area_rmse)
    print('mean: ', mean_rmse)

    save_data(X_train, area_oof, area_pred, rmse=area_rmse,
              name=f'area_seed_{i}', save_dir='./seed_average')
