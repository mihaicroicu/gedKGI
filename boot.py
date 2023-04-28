import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import click

np.random.seed(403961)

@click.command()
@click.option('--iter',default=1,help='MonteCarlo Iteration ID.')
@click.option('--kgi',default=0.1,help='KGI level.')
@click.option('--bias',default=0.1,help='Bias.')
def read_opt(iter, bias, kgi):
    """Read options for training KGI models across MC simulations."""
    return iter,kgi,bias


if __name__ == "__main__":
    ii,k,b = read_opt(standalone_mode=False)
    print(f'k:{k}, b:{b}, i:{ii}')
    if b==0:
        b='0'
    df = pd.read_parquet(f'storage/kgi_pred_results/results_{ii}_kgi{k}_bias{b}.parquet')

    out_metrics = []

    rmse_k = np.sqrt(mean_squared_error(y_true=df['actuals'],y_pred=df['preds_k']))
    rmse_l = np.sqrt(mean_squared_error(y_true=df['actuals'],y_pred=df['preds_l']))

    dftrim = df[df['preds_k'] != float(df['preds_k'].mode())]
    trim_k = np.sqrt(mean_squared_error(y_true=dftrim['actuals'],y_pred=dftrim['preds_k']))
    dftrim_w = df[df['preds_l'] != float(df['preds_l'].mode())]
    trim_l = np.sqrt(mean_squared_error(y_true=dftrim_w['actuals'],y_pred=dftrim_w['preds_l']))


    for i in range(50):
        print(i)
        model = np.random.choice(df.columns[df.columns.str.contains('preds_i')],
                         100,replace=True)

        df['t_ens'] = df[model].mean(axis=1)

        rmsle_full = np.sqrt(mean_squared_error(y_true=df['actuals'],
                                                y_pred=df['t_ens']))

        dftrim = df[df['t_ens'] != float(df['t_ens'].mode())]

        rmsle_trim = np.sqrt(mean_squared_error(y_true=dftrim['actuals'],
                                                y_pred=dftrim['t_ens']))

        out_metrics+=[(rmsle_full, rmsle_trim)]

    boot_measures = pd.DataFrame(out_metrics,columns=['i_full','i_trimmed'])

    boot_measures['k_full']=rmse_k
    boot_measures['l_full']=rmse_l

    boot_measures['k_trim']=trim_k
    boot_measures['l_trim']=trim_l

    dfmin =  df[df['preds_k'] != float(df['preds_k'].mode())]
    dfmin =  dfmin[dfmin['preds_l'] != float(dfmin['preds_l'].mode())]
    dfmin =  dfmin[dfmin['t_ens'] != float(dfmin['t_ens'].mode())]
    trim_k = np.sqrt(mean_squared_error(y_true=dfmin['actuals'],y_pred=dfmin['preds_k']))
    trim_l = np.sqrt(mean_squared_error(y_true=dfmin['actuals'],y_pred=dfmin['preds_l']))


    boot_measures['l_min']=trim_l
    boot_measures['k_min']=trim_k

    boot_measures['kgi']=k
    boot_measures['bias']=b
    boot_measures['iter']=ii
    print(boot_measures.head())
    path = f'storage/kgi_pred_results/boot_measures_{ii}_kgi{k}_bias{b}.parquet'
    print (f"Saving to {path}")
    boot_measures.to_parquet(path)
