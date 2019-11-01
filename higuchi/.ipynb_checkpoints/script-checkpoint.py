from collections import Counter
from catboost import CatBoost
from catboost import Pool
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import sys
from sklearn.ensemble import RandomForestRegressor
from time import time
import datetime
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
'''
使い方
lgbPredictor = RegressionPredictor(train_df, train_y, test_df, params=lgbm_params,
                                   sk_model=None, n_splits=10, clf_type='lgb')
lgboof, lgbpreds, lgbFIs = lgbPredictor.fit()
lgbPredictor.plot_FI(50)
lgbPredictor.plot_pred_dist()

'''


class RegressionPredictor(object):
    '''
    回帰をKfoldで学習するクラス。
    X->dataframe,y->numpy.array
    TODO:分類、多クラス対応/Folderを外部から渡す/predictのプロット/できれば学習曲線のプロット
    '''

    def __init__(self, train_X, train_y, test_X, params=None, Folder=None, sk_model=None, n_splits=5, clf_type='xgb', aggfunc_dict=None, verbose_eval=5000):
        self.kf = Folder if Folder !=None else KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.columns = train_X.columns.values
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.verbose_eval = verbose_eval
        self.params = params
        self.aggfunc_dict = aggfunc_dict
        TE_cols = np.sum([len(x) for x in self.aggfunc_dict.values()]) if aggfunc_dict != None else 0
        self.oof = np.zeros((len(self.train_X),))
        self.preds = np.zeros((len(self.test_X),))
        if clf_type == 'xgb':
            self.FIs = {}
        else:
            self.FIs = np.zeros(self.train_X.shape[1] + TE_cols, dtype=np.float)
        self.sk_model = sk_model
        self.clf_type = clf_type
        self.aggfunc_dict = aggfunc_dict

    @staticmethod
    def merge_dict_add_values(d1, d2):
        return dict(Counter(d1) + Counter(d2))

    def rmse(self):
        return int(np.sqrt(mean_squared_error(self.oof, self.train_y)))

    def get_model(self):
        return self.model

    def target_encoding(self, tr_X, val_X, tr_y):
        '''CVとTEのfoldを同じにした。
        test,valのTEはtr_Xの情報のみで作られる。
        self.testが更新されるが、毎回のCVで上書きされる
        TODO:後からTEを取り出せるようにする。
        '''
        tr_X['target'] = tr_y
        for category_col, func_list in self.aggfunc_dict.items():
            agg_func = {'target': func_list}
            #agg_funcでgruopby
            agg_df = tr_X.groupby(category_col)['target'].agg(agg_func)
            #列名作成
            agg_df.columns = [category_col + '_' + '_'.join(col).strip() for col in agg_df.columns.values]

            #元の列に集約結果をmapしその値をコピーし新規列に加え返す。
            #列名を追加する。
            if not set(agg_df.columns.values).issubset(set(self.columns)):
                self.columns = np.append(self.columns, agg_df.columns.values)
            for col in agg_df.columns.values:
                tr_X[col] = tr_X[category_col].map(agg_df[col]).copy()
                val_X[col] = val_X[category_col].map(agg_df[col]).copy()
                self.test_X[col] = self.test_X[category_col].map(agg_df[col]).copy()
        tr_X.drop(columns='target', inplace=True)
        return tr_X, val_X

    def _get_xgb_callbacks(self):
        '''nround,early_stopをparam_dictから得るためのメソッド'''
        nround = 1000
        early_stop_rounds = 10
        if self.params['num_boost_round']:
            nround = self.params['num_boost_round']
            del self.params['num_boost_round']

        if self.params['early_stopping_rounds']:
            early_stop_rounds = self.params['early_stopping_rounds']
            del self.params['early_stopping_rounds']
        return nround, early_stop_rounds

    def _get_cv_model(self, tr_X, val_X, tr_y, val_y, val_idx):

        if self.clf_type == 'cat':
            clf_train = Pool(tr_X, tr_y)
            clf_val = Pool(val_X, val_y)
            clf_test = Pool(self.test_X)
            self.model = CatBoost(params=self.params)
            self.model.fit(clf_train, eval_set=[clf_val])
            self.oof[val_idx] = self.model.predict(clf_val)
            self.preds += self.model.predict(clf_test) / self.kf.n_splits
            self.FIs += self.model.get_feature_importance()

        elif self.clf_type == 'lgb':
            clf_train = lgb.Dataset(tr_X, tr_y)
            clf_val = lgb.Dataset(val_X, val_y, reference=lgb.train)
            self.model = lgb.train(self.params, clf_train, valid_sets=[clf_train, clf_val], verbose_eval=self.verbose_eval)
            self.oof[val_idx] = self.model.predict(val_X, num_iteration=self.model.best_iteration)
            self.preds += self.model.predict(self.test_X, num_iteration=self.model.best_iteration) / self.kf.n_splits
            self.FIs += self.model.feature_importance(importance_type='gain')

        elif self.clf_type == 'xgb':
            clf_train = xgb.DMatrix(tr_X, label=tr_y, feature_names=self.columns)
            clf_val = xgb.DMatrix(val_X, label=val_y, feature_names=self.columns)
            clf_test = xgb.DMatrix(self.test_X, feature_names=self.columns)
            evals = [(clf_train, 'train'), (clf_val, 'eval')]
            evals_result = {}

            nround, early_stop_rounds = self._get_xgb_callbacks()
            self.model = xgb.train(self.params,
                                   clf_train,
                                   num_boost_round=nround,
                                   early_stopping_rounds=early_stop_rounds,
                                   verbose_eval=self.verbose_eval,
                                   evals=evals,
                                   evals_result=evals_result)

            self.oof[val_idx] = self.model.predict(clf_val)
            self.preds += self.model.predict(clf_test) / self.kf.n_splits
            self.FIs = self.merge_dict_add_values(self.FIs, self.model.get_fscore())

        elif self.clf_type == 'sklearn':
            self.model = self.sk_model
            self.model.fit(tr_X, tr_y)
            self.oof[val_idx] = self.model.predict(val_X)
            self.preds += self.model.predict(self.test_X) / self.kf.n_splits
            self.FIs += self.model.feature_importances_
        else:
            raise ValueError('clf_type is wrong.')

    def fit(self):
        start_time = time()

        for i, (train_idx, val_idx) in enumerate(self.kf.split(self.train_X, self.train_y)):

            print(f'Training on fold {i+1}')
            X_train = self.train_X.iloc[train_idx, :]
            X_val = self.train_X.iloc[val_idx, :]
            y_train = self.train_y[train_idx]
            y_val = self.train_y[val_idx]
            if self.aggfunc_dict != None:
                X_train, X_val = self.target_encoding(X_train, X_val, y_train)
            self._get_cv_model(X_train, X_val, y_train, y_val, val_idx)

        print('-' * 30)
        print('Training has finished.')
        print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - start_time))))
        print('-' * 30)
        print('this self.model`s rmse:', self.rmse())

        return self.oof, self.preds, self.FIs

    def plot_FI(self, max_row=50):
        plt.figure(figsize=(10, 20))
        if self.clf_type == 'xgb':
            df = pd.DataFrame.from_dict(self.FIs, orient='index').reset_index()
            df.columns = ['col', 'FI']
        else:
            df = pd.DataFrame({'FI': self.FIs, 'col': self.columns})
        df = df.sort_values('FI', ascending=False).reset_index(drop=True).iloc[:max_row, :]
        sns.barplot(x='FI', y='col', data=df)
        plt.show()

    def plot_pred_dist(self):
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        sns.distplot(self.oof, ax=axs[0], label='oof')
        sns.distplot(self.train_y, ax=axs[0], label='train_y')
        axs[0].legend(labels=['oof', 'train_y'])
        sns.distplot(self.preds, ax=axs[1], label='test_preds')
        plt.show()
    
    def plot_scatter_oofvspred(self):
        print('RMSE : ', np.sqrt(mean_squared_error(self.oof, self.train_y)))
        print('R^2 : ', r2_score(self.oof, self.train_y))
        plt.figure()
        plt.scatter(self.oof, self.train_y, alpha=0.7)
        plt.title("$R^2 = {:<.5}$".format(r2_score(self.oof, self.train_y)))
        x = np.linspace(0, 2500000, 100)
        plt.plot(x, x, c="indianred")
        plt.xlabel('predict', size=20)
        plt.ylabel('correct', size=20)


class LogRegressionPredictor(RegressionPredictor):
    '''学習時に目的変数を変換するときに使う。例えばlogを取ってから学習する場合など。
         revertには逆操作となる関数を定義する。    
    '''
    def __init__(self, train_X, train_y, test_X, params=None, Folder=None, sk_model=None, n_splits=5, clf_type='xgb', aggfunc_dict=None, verbose_eval=5000, func=np.log1p, revert_func=np.expm1):
        self.func=func
        self.revert_func=revert_func
        super(LogRegressionPredictor, self).__init__(train_X, self.func(train_y), test_X, params,
                                                     Folder, sk_model, n_splits, clf_type, aggfunc_dict, verbose_eval)

    def _get_cv_model(self, tr_X, val_X, tr_y, val_y, val_idx):

        if self.clf_type == 'cat':
            clf_train = Pool(tr_X, tr_y)
            clf_val = Pool(val_X, val_y)
            clf_test = Pool(self.test_X)
            self.model = CatBoost(params=self.params)
            self.model.fit(clf_train, eval_set=[clf_val])
            self.oof[val_idx] = self.revert_func(self.model.predict(clf_val))
            self.preds += self.revert_func(self.model.predict(clf_test)) / self.kf.n_splits
            self.FIs += self.model.get_feature_importance()

        elif self.clf_type == 'lgb':
            clf_train = lgb.Dataset(tr_X, tr_y)
            clf_val = lgb.Dataset(val_X, val_y, reference=lgb.train)
            self.model = lgb.train(self.params, clf_train, valid_sets=[
                                   clf_train, clf_val], verbose_eval=self.verbose_eval)
            self.oof[val_idx] = self.revert_func(self.model.predict(val_X, num_iteration=self.model.best_iteration))
            self.preds += self.revert_func(self.model.predict(self.test_X, num_iteration=self.model.best_iteration)) / self.kf.n_splits
            self.FIs += self.model.feature_importance(importance_type='gain')

        elif self.clf_type == 'xgb':
            clf_train = xgb.DMatrix(tr_X, label=tr_y, feature_names=self.columns)
            clf_val = xgb.DMatrix(val_X, label=val_y, feature_names=self.columns)
            clf_test = xgb.DMatrix(self.test_X, feature_names=self.columns)
            evals = [(clf_train, 'train'), (clf_val, 'eval')]
            evals_result = {}

            nround, early_stop_rounds = self._get_xgb_callbacks()
            self.model = xgb.train(self.params,
                                   clf_train,
                                   num_boost_round=nround,
                                   early_stopping_rounds=early_stop_rounds,
                                   verbose_eval=self.verbose_eval,
                                   evals=evals,
                                   evals_result=evals_result)

            self.oof[val_idx] = self.revert_func(self.model.predict(clf_val))
            self.preds += self.revert_func(self.model.predict(clf_test)) / self.kf.n_splits
            self.FIs = self.merge_dict_add_values(self.FIs, self.model.get_fscore())

        elif self.clf_type == 'sklearn':
            self.model = self.sk_model
            self.model.fit(tr_X, tr_y)
            self.oof[val_idx] = self.revert_func(self.model.predict(val_X))
            self.preds += self.revert_func(self.model.predict(self.test_X)) / self.kf.n_splits
            self.FIs += self.model.feature_importances_
        else:
            raise ValueError('clf_type is wrong.')

    def plot_pred_dist(self):
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        sns.distplot(self.oof, ax=axs[0], label='oof')
        sns.distplot(self.revert_func(self.train_y), ax=axs[0], label='train_y')
        axs[0].legend(labels=['oof', 'train_y'])
        sns.distplot(self.preds, ax=axs[1], label='test_preds')
        plt.show()
    
    def plot_scatter_oofvspred(self):
        print('RMSE : ', np.sqrt(mean_squared_error(self.oof, self.revert_func(self.train_y))))
        print('R^2 : ', r2_score(self.oof, self.revert_func(self.train_y)))
        plt.figure()
        plt.scatter(self.oof, self.revert_func(self.train_y), alpha=0.7)
        plt.title("$R^2 = {:<.5}$".format(r2_score(self.oof, self.revert_func(self.train_y))))
        x = np.linspace(0, 2500000, 100)
        plt.plot(x, x, c="indianred")
        plt.xlabel('predict', size=20)
        plt.ylabel('correct', size=20)
    
    def rmse(self):
        return int(np.sqrt(mean_squared_error(self.oof, self.revert_func(self.train_y))))
