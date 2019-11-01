import pandas as pd
from pathlib import Path

def save_data(oof,preds,rmse=20000,name='lgb',submit_dir='./submit',save_dir='./submit'):
    '''stacking用にtrain,testの予測値を同じ形で保存する
    from pathlib import Path　がいる
    '''
    submit_dir=Path(submit_dir)
    submit_path=submit_dir/'sample_submit.csv'
    train_id_path=submit_dir/'train_id.csv'
    pred_path=save_dir/f'submit_{rmse}_{name}.csv'
    oof_path=save_dir/f'oof_{rmse}_{name}.csv'
    
    submit = pd.read_csv(submit_path,header=None).rename(columns={0:'id',1:'target'})
    train_oof=pd.read_csv(train_id_path,header=None)
    submit['target'] = preds
    train_oof['target']=oof
    submit.to_csv(pred_path, index=None, header=None)
    train_oof.to_csv(oof_path, index=None, header=None)
    
    display(submit.head())
    display(train_oof.head())
    
   
def making_train_id_df(train,path='./submit/train_id.csv'):
    '''使ったtrain_idを保持しておくDataFrameを作成する'''
    train_id_df=pd.DataFrame()
    train_id_df['id']=train['id'].values
    train_id_df['traget']=0
    train_id_df.to_csv(path,header=None,index=None)