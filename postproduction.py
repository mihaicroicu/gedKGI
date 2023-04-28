import click
import os
import lightgbm as lgbm  
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from kgi_point import ADMShapes, KGIPoint
from shapely.geometry import Polygon, Point
from mingester.extensions import *

import warnings

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


def rmse_col(df, col):
    return np.expm1(np.sqrt(mean_squared_error(y_true = df['actuals'], y_pred=df[col])))

def centroid2pg(row):
    centroid = KGIPoint(row['latitude'],row['longitude'],6,cshp).polygon.centroid
    return Priogrid.from_lat_lon(lon=centroid.x, lat=centroid.y).id

def select_imp(run_id):
    x = kgi_imp[['id','month_id','best',run_id]].rename(columns={run_id:'pg_id'})
    print(x.head(10), x.shape)
    return x

def splag0(id):
    return id
def splag(id):
    #print(type(id))
    if type(id)==int or type(id)==np.int64:
        return list([i.id for i in list(np.concatenate(Priogrid(id).queen_contiguity()).flat)])
    if type(id)==list:
        return list(np.unique(list(np.concatenate([splag(i) for i in id]).flat)))
def splag2(id): return splag(splag(id))

def df_splag(df, func):
    kgi_lag1 = df.copy()
    kgi_lag1['pg_id'] = kgi_lag1.pg_id.apply(func)
    rfd = kgi_lag1.explode('pg_id')[['month_id','best','pg_id']].groupby(['month_id','pg_id']).sum().reset_index()
    return rfd

def tlag(df, months=1):
    rdf = df.copy()
    rdf['month_id'] = rdf['month_id']+months
    return rdf

def make_smpl(df):
    base_df = df.copy()
    sp0 = df_splag(base_df,splag0).set_index(['month_id','pg_id'])
    sp1 = df_splag(base_df,splag).set_index(['month_id','pg_id']).rename(columns={'best':'best_sp1'})
    sp2 = df_splag(base_df,splag2).set_index(['month_id','pg_id']).rename(columns={'best':'best_sp2'})
    spatial = sp0.join([sp1,sp2],how='outer').fillna(0)
    act = sp0.reset_index().copy()
    leads=[-3,-6,-12]
    leads = [tlag(act,k).rename(columns={'best':f'y{-k}'}).set_index(['month_id','pg_id']) for k in leads]
    tmp = []
    for i in range(1,13):
        df =spatial.copy().reset_index()
        df=tlag(df,i).rename(columns={'best':f't{i}','best_sp1':f't{i}_sp1','best_sp2':f't{i}_sp2'})
        tmp+=[df.set_index(['month_id','pg_id'])]
    full = spatial.join(tmp, how='outer').join(leads,how='outer').fillna(0).reset_index()
    full = full[full.month_id>=108]
    full = full.pgm.fill_bbox().fillna(0)
    return full

def train_test(df):
    y = df[f'y{steps}']
    X = df.copy()
    del X[f'y{steps}']
    return X,y

def mk_train_test(df, steps=3):
    z = df.copy()
    z.month_id = z.month_id - steps
    for i in [3,6,12]:
        if i!=steps:
            del z[f'y{i}']
    z[f'y{steps}'] = np.log1p(z[f'y{steps}'])
    train = z[z.month_id.between(110,444)]
    test = z[z.month_id.between(445,504)]
    return train,test

def splitXY(train, test, steps=3):
    
    ycol = f'y{steps}'
    
    y_train = train[ycol]
    X_train = train.copy()
    del X_train['pg_id']#,'month_id','y3']
    del X_train['month_id']
    del X_train[ycol]
    #X_train

    y_test = test[ycol]
    X_test = test.copy()
    del X_test['pg_id']#,'month_id','y3']
    del X_test['month_id']
    del X_test[ycol]
    #X_test
    return X_train, X_test, y_train, y_test


@click.command()
@click.option('--seed',default=42,help='MonteCarlo Iteration ID.')
@click.option('--iter',default=1,help='MonteCarlo Iteration ID.')
@click.option('--kgi',default=0.1,help='KGI level.')
@click.option('--bias',default=0.1,help='Bias.')
@click.option('--prec',default=6,help='Where_prec.')
def read_opt(seed, iter, bias, kgi, prec):
    """Read options for training KGI models across MC simulations."""
    #print(f"{seed=}, {iter=}, {kgi=}, {bias=}, {prec=}")
    return seed,iter,kgi,bias, prec

if __name__ == "__main__":
    version = '22.1'
    seed,iter,kgi,bias,prec=read_opt(standalone_mode=False)
    print(f"""Running the KGI prediction-based analysis with: 
            {seed=}, {iter=}, {kgi=}, {bias=}""")

    path = 'storage/ged/'
    out_path = 'storage/output/'
    if bias == 0:
       bias = '0'
    command = "ls " + out_path + f" |  grep dyad_ | grep _{iter}_ | grep kgi{kgi} | grep bias{bias}.par | " + "awk -F '_' '{print $2}' | uniq"
    dyads = os.popen(command).read()
    dyads = [int(i) for i in dyads.split("\n") if i != '']
    dyads = sorted(dyads)
    print(f'{dyads=} totalling {len(dyads)}')

    mc_round = iter
    list_conf = dyads

    if bias==0:
        bias='0'

    train = pd.read_parquet(f'{path}biter_{mc_round}_train_kgi{kgi}_bias{bias}.parquet')
    test = pd.read_parquet(f'{path}biter_{mc_round}_test_kgi{kgi}_bias{bias}.parquet')
    train = train[train.dyad_new_id.isin(list_conf)]
    test = test[test.dyad_new_id.isin(list_conf)]

    fatalities = pd.concat([train[['id','month_id','best']],test[['id','month_id','best']]], ignore_index=True)

    tt = train[['id','month_id','priogrid_gid']]
    tt['key'] = 0
    mr = pd.DataFrame({'key':0, 'method':list(range(1000,1050))+list(range(2050,2100))})
    tt = tt.merge(mr, on=['key'])[['id','method','month_id','priogrid_gid']].rename(columns={'priogrid_gid':'pg_id'})

    # Prepare the KGI imputation event positionings (100 datasets)

    print("Preparing GED datasets with KGI...", flush=True)

    mc_df = pd.DataFrame()
    for dyad in list_conf:
        df = pd.read_parquet(f'{out_path}dyad_{dyad}_{mc_round}_kgi{kgi}_bias{bias}.parquet')
        df['method'] = df.method*1000 + df.index % 100
        df = pd.DataFrame.pg.from_latlon(df,lat_col='latitude',lon_col='longitude')[['id','method','pg_id']]
        mc_df = pd.concat([mc_df,df],ignore_index=True)
    print(mc_df.head(10))
    print("MC_DF")
    mc_df = pd.concat([mc_df,tt],ignore_index=True)
    print(mc_df.head(100))
    print("KGI_IMP")
    kgi_imp = pd.pivot_table(mc_df, values='pg_id', index=['id'], columns=['method']).reset_index()
    print(kgi_imp.head(9))
    print("HED_IMP")
    kgi_imp = fatalities.merge(kgi_imp, on='id', how='outer')
    print (kgi_imp.head(10))
    print (kgi_imp[kgi_imp.id==1489])
    #exit(1)

    print("Preparing list deletions...")
    list_dele = train[['id','month_id','best','priogrid_gid']].rename(columns={'priogrid_gid':'pg_id'})

    print("Preparing UCDP-style imputations...")
    cshp = ADMShapes("storage/afr_cntry.gpkg")
    ucdp_kgi = test.copy()
    ucdp_kgi['pg_id'] = test.apply(centroid2pg, axis=1)
    ucdp_kgi = pd.concat([list_dele,ucdp_kgi[['id','month_id','best','pg_id']]],ignore_index=True)

    base_df = ucdp_kgi.copy()

    print ("Building simple feature set for forecasting model...")
    df_k = make_smpl(ucdp_kgi)
    print ("Building simple feature set for forecasting model...")
    df_l = make_smpl(list_dele)

    datasets = {'k':df_k,'l':df_l}

    print("Forecasting! Please wait...", flush=True)
    results = []
    for k,vdf in datasets.items():
        print(f"{k=}")
        train, test = mk_train_test(vdf,3)
        X_train, X_test, y_train, y_test = splitXY(train, test, 3)

        reg = lgbm.LGBMRegressor(n_estimators=500)
        eval_set = [(X_test, y_test)]

        model=reg.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=15,
        )

        collector = test[['pg_id','month_id']]
        collector[f'preds_{k}'] = np.expm1(reg.predict(X_test))
        results += [collector.set_index(['pg_id','month_id'])]

    print("Making feature sets for KGI forecasting model", flush=True)
    tries = list(range(1000,1050))+list(range(2050,2100))
    for i in tries:
        print(i)
        cur_kgi_dset = make_smpl(select_imp(i))
        train, test = mk_train_test(cur_kgi_dset, 3)
        X_train, X_test, y_train, y_test = splitXY(train, test, 3)
        reg = lgbm.LGBMRegressor(n_estimators=500)
        eval_set = [(X_test, y_test)]

        model = reg.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=15,
        )

        collector = test[['pg_id', 'month_id']]
        collector[f'preds_i{i}'] = np.expm1(reg.predict(X_test))
        results += [collector.set_index(['pg_id', 'month_id'])]

    cur_kgi_dset = None
    print("Forecasting complete...", flush=True)

    results = pd.DataFrame().join(results, how='outer')
    results['i_avg'] = results[results.columns[results.columns.str.contains('_i')]].mean(axis=1)
    test_fat = pd.concat([train[['pg_id','month_id','best']],test[['pg_id','month_id','best']]], ignore_index=True)
    test_fat = test_fat.groupby(['pg_id','month_id']).sum().rename(columns={'best':'actuals'})
    results = results.join(test_fat, how='left')
    #Due to how the spatial lags are computed, edges may result in different resulting bboxes. 
    results = results.dropna()
    log_results = np.log1p(results)

    rmse_i = np.expm1(np.sqrt(mean_squared_error(y_true = log_results['actuals'], y_pred=log_results['i_avg'])))
    rmse_l = np.expm1(np.sqrt(mean_squared_error(y_true = log_results['actuals'], y_pred=log_results['preds_l'])))
    rmse_k = np.expm1(np.sqrt(mean_squared_error(y_true = log_results['actuals'], y_pred=log_results['preds_k'])))
    rmse_i = rmse_col(log_results,'i_avg')
    rmse_s1 = rmse_col(log_results,'preds_i1003')
    rmse_s2 = rmse_col(log_results,'preds_i2098')
    rmse_l = rmse_col(log_results,'preds_l')
    rmse_k = rmse_col(log_results,'preds_k')

    agg_data = pd.DataFrame({'iter':[mc_round],'type':['full'],
                'rmse_i':[rmse_i],'rmse_s1':[rmse_s1],'rmse_s2':[rmse_s2],
                'rmse_k':[rmse_k],'rmse_l':[rmse_l]})

    #print("RLMSE for full data:")
    #print(agg_data)

    trimmed_res = log_results.copy()
    trimmed_res = trimmed_res[trimmed_res.preds_k != float(trimmed_res.preds_k.mode())]
    trimmed_res = trimmed_res[trimmed_res.preds_l != float(trimmed_res.preds_l.mode())]
    trimmed_res = trimmed_res[trimmed_res.i_avg != float(trimmed_res.i_avg.mode())]
    rmse_i = rmse_col(trimmed_res,'i_avg')
    rmse_s1 = rmse_col(trimmed_res,'preds_i1003')
    rmse_s2 = rmse_col(trimmed_res,'preds_i2098')
    rmse_l = rmse_col(trimmed_res,'preds_l')
    rmse_k = rmse_col(trimmed_res,'preds_k')

    agg_data2 = pd.DataFrame({'iter':[mc_round],'type':['full'],
                'rmse_i':[rmse_i],
                'rmse_s1':[rmse_s1],'rmse_s2':[rmse_s2],
                'rmse_k':[rmse_k], 'rmse_l':[rmse_l]})

    print("RMLSE results:")
    print("*"*24)
    #print(agg_data)

    print("Storing data:")
    pd.concat([agg_data,agg_data2]).to_csv(f'storage/kgi_pred_results/measures_{mc_round}_kgi{kgi}_bias{bias}.parquet.csv')
    print(pd.concat([agg_data,agg_data2]))
    log_results.to_parquet(f'storage/kgi_pred_results/results_{mc_round}_kgi{kgi}_bias{bias}.parquet')

    print("Done")
