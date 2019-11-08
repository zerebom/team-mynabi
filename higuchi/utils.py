import pandas as pd
from pathlib import Path
import datetime
def save_data(train,oof,preds,rmse=20000,name='lgb',submit_dir='./submit',save_dir='./submit'):
    '''stacking用にtrain,testの予測値を同じ形で保存する
    from pathlib import Path　がいる
    predict.csvの情報を記載したcsvも作成する
    '''
    submit_dir=Path(submit_dir)
    save_dir=Path(save_dir)
    submit_path=submit_dir/'sample_submit.csv'
    train_id_path=submit_dir/'train_id.csv'
    pred_path=save_dir/f'submit_{rmse}_{name}.csv'
    oof_path=save_dir/f'oof_{rmse}_{name}.csv'
    train_log_path=save_dir/f'pred_log.csv'
    
    submit = pd.read_csv(submit_path,header=None).rename(columns={0:'id',1:'target'})
    train_oof=pd.read_csv(train_id_path,header=None).rename(columns={0:'id',1:'target'})
    submit['target'] = preds
    train_oof['target']=oof
    submit.to_csv(pred_path, index=None, header=None)
    train_oof.to_csv(oof_path, index=None, header=None)
    
    if train_log_path.is_file():
        train_log_df=pd.read_csv(train_log_path)
    else:
        train_log_df=pd.DataFrame(columns=['path','features','rmse','time'])
    s=pd.Series([pred_path,','.join(train.columns),rmse,str(datetime.date.today())],index=['path','features','rmse','time'],name=train_log_df.shape[0])
    train_log_df=train_log_df.append(s)
    train_log_df.to_csv(train_log_path,index=None)
    
    #display(submit.head())
    #display(train_oof.head())
    
def making_train_id_df(train,path='./submit/train_id.csv'):
    '''使ったtrain_idを保持しておくDataFrameを作成する'''
    train_id_df=pd.DataFrame()
    train_id_df['id']=train['id'].values
    train_id_df['traget']=train['賃料'].values
    train_id_df.to_csv(path,header=None,index=None)
    

