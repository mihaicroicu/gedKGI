import pandas as pd
import numpy as np
import click
from sklearn.metrics import mean_squared_error



def rmse_col(df, col):
    return np.expm1(np.sqrt(mean_squared_error(y_true = df['actuals'], y_pred=df[col])))


@click.command()
@click.option('--iter',default=99,help='MonteCarlo Iteration ID.')
@click.option('--kgi',default=0.1,help='KGI level.')
@click.option('--bias',default=0.1,help='Bias.')
@click.option('--esize',default=50,help='Ensemble size.')
@click.option('--pm2',default=.5,help='Proportion of the M2 models.')
def read_opt(iter, bias, kgi, pm2, esize):
    """Read options for training KGI models across MC simulations."""
    return iter,kgi,bias,esize,pm2


if __name__ == "__main__":
    iter,kgi,bias,esize,pm2 = read_opt(standalone_mode=False)
    res_path = 'storage/kgi_pred_results/'
    out_path = 'storage/ens/'
    #iter = 22
    #kgi=.1
    #bias=.3
    size = esize
    m2_split_perc = pm2

    if bias==0: bias='0'

    measures = pd.read_csv(f'{res_path}measures_{iter}_kgi{kgi}_bias{bias}.parquet.csv')
    log_results =  pd.read_parquet(f'{res_path}results_{iter}_kgi{kgi}_bias{bias}.parquet')

    m1 = [f'preds_i{i}' for i in list(range(1000,1050))]
    m2 = [f'preds_i{i}' for i in list(range(2050,2100))]

    m2_size = int(np.round(size*m2_split_perc))
    m1_size = size-m2_size

    print(m1_size, m2_size)

    np.random.seed(42)
    out_res = []
    out_res_trimmed = []
    for i in range(1000):
        print(i,':')
        v_set1 = np.random.choice(m1,size=m1_size, replace=True)
        v_set2 = np.random.choice(m2,size=m2_size, replace=True)
        v_set = list(v_set1)+list(v_set2)
        print(v_set)
        log_results['tmp'] = log_results[v_set].mean(axis=1)
        trimmed_res = log_results.copy()
        trimmed_res = trimmed_res[trimmed_res.tmp != float(trimmed_res.tmp.mode())]
        out_res += [rmse_col(log_results,'tmp')]
        out_res_trimmed += [rmse_col(trimmed_res,'tmp')]

    out = pd.DataFrame(out_res, columns=['i'])
    out['k'] = rmse_col(log_results,'preds_k')
    out['l'] = rmse_col(log_results,'preds_l')
    out.to_parquet(f'{out_path}/ens_size{size}_split{m2_split_perc}_iter{iter}_kgi{kgi}_bias{bias}.parquet')
    
    out = pd.DataFrame(out_res_trimmed, columns=['i'])
    out['k'] = rmse_col(trimmed_res,'preds_k')
    out['l'] = rmse_col(trimmed_res,'preds_l')
    out.to_parquet(f'{out_path}/trimmed_size{size}_split{m2_split_perc}_iter{iter}_kgi{kgi}_bias{bias}.parquet')
    print("Done")





