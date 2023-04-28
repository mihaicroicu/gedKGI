import click
import pandas as pd
import numpy as np
import tensorflow as tf

import gpflow
from gpflow.utilities import print_summary

from typing import Union

from kgi import KGIgpp, decay_ged, ged_from_file
from kgi_point import ADMShapes, KGIPoint

from shapely.geometry import Polygon, Point

from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


tf.random.set_seed(24582)
np.random.seed(79933)

def process_row(row: pd.DataFrame, kgi1: KGIgpp, kgi2: KGIgpp) -> pd.DataFrame:
    kgi_point = KGIPoint(row.latitude,row.longitude,6,cshp)
    kgi_sample1 = kgi_point.sample_gdf(kgi1, n=50)
    kgi_sample2 = kgi_point.sample_gdf(kgi2, n=50)

    kgi_sample1 = pd.DataFrame({'id':row.id, 'method':1,
                  'latitude':kgi_sample1.geometry.y, 
                  'longitude':kgi_sample1.geometry.x
                  })

    kgi_sample2 = pd.DataFrame({'id':row.id, 'method':2,
                  'latitude':kgi_sample2.geometry.y, 
                  'longitude':kgi_sample2.geometry.x
                  })

    kgi_sample = pd.concat([kgi_sample1,kgi_sample2],ignore_index=True)
    return kgi_sample

def make_run(row):
    print (row.dyad_new_id, row.month_id)
    train_d = train[(train.dyad_new_id==row.dyad_new_id) & (train.month_id.between(row.month_id-24,row.month_id))]
    train_d['best']=decay_ged(train_d)
    train_d = train_d[['latitude','longitude','best']].groupby(['latitude','longitude']).sum()
    train_d = np.log1p(train_d)

    train_d = train_d[train_d.best>0].reset_index()
    train_d['best'] = train_d['best'].astype('float')
    train_d

    test_d = test[(test.dyad_new_id==row.dyad_new_id) & (test.month_id==row.month_id)]

    X = train_d[['latitude','longitude']]
    y = train_d[['best']]

    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=0.25, variance=0.5)
    kernel1.lengthscales.prior = tfp.distributions.LogNormal(tf.math.log(gpflow.utilities.to_default_float(.25)), .25)
    kernel1.variance.prior = tfp.distributions.LogNormal(tf.math.log(gpflow.utilities.to_default_float(np.mean(train_d.best))), 10)

    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=0.5, variance=1)
    kernel2.lengthscales.prior = tfp.distributions.LogNormal(tf.math.log(gpflow.utilities.to_default_float(.5)), .5)
    kernel2.variance.prior = tfp.distributions.LogNormal(tf.math.log(gpflow.utilities.to_default_float(np.mean(train_d.best))), 10)

    kgi1 = KGIgpp(X=X, y=y, kernel=kernel1)
    kgi1.train()
    #preds = kgi1.predict_gdf()

    kgi2 = KGIgpp(X=X, y=y, kernel=kernel2)
    kgi2.train()
    #preds = kgi2.predict_gdf()

    process_this = partial(process_row ,kgi1=kgi1, kgi2=kgi2)
    out = test_d.apply(process_this, axis=1)
    out = pd.concat([*out],ignore_index=True)
    out['id'] = out['id'].astype('int')
    return out


@click.command()
@click.option('--seed',default=42,help='MonteCarlo Iteration ID.')
@click.option('--iter',default=1,help='MonteCarlo Iteration ID.')
@click.option('--kgi',default=0.1,help='KGI level.')
@click.option('--bias',default=0.1,help='Bias.')
@click.option('--prec',default=6,help='KGI at level 4 (ADM) or 6 (CNTRY)?')
@click.option('--dyad',default=None, help='Dyad')
def read_opt(seed, iter, bias, kgi, prec, dyad):
    """Read options for training KGI models across MC simulations."""
    #print(f"{seed=}, {iter=}, {kgi=}, {bias=}, {prec=}")
    return seed,iter,kgi,bias, prec, dyad

if __name__ == '__main__':
    version = '22.1'
    path = 'storage/ged/'
    seed,iter,kgi,bias,prec, dyad=read_opt(standalone_mode=False)
    print(f"{seed=}, {iter=}, {kgi=}, {bias=}, {dyad=}")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    print(f'Fetching: {path}biter_{iter}_train_kgi{kgi}_bias{bias}.parquet')

    if bias == 0: bias='0'

    train = pd.read_parquet(f'{path}biter_{iter}_train_kgi{kgi}_bias{bias}.parquet')
    test = pd.read_parquet(f'{path}biter_{iter}_test_kgi{kgi}_bias{bias}.parquet')

    dyad_todo_list = test[['dyad_new_id','month_id']].drop_duplicates().reset_index(drop=True).\
        sort_values(by=['dyad_new_id','month_id'])

    if dyad is not None:
        dyad = int(dyad)
        print (f"Only doing dyads larger than : {dyad}")
        dyad_todo_list = dyad_todo_list[dyad_todo_list.dyad_new_id==dyad]

    adm1 = ADMShapes("storage/afr_adm1.gpkg")
    cshp = ADMShapes("storage/afr_cntry.gpkg")

    print(f"To do: {dyad_todo_list.shape[0]} dyad months")

for dyad in dyad_todo_list.dyad_new_id.unique():
    print(f'Now doing {dyad=}')
    run_list = dyad_todo_list[dyad_todo_list.dyad_new_id==dyad]
    out = run_list.apply(make_run,axis=1)
    out_dat = pd.concat([*out],ignore_index=True)
    out_dat
    out_dat.to_parquet(f'storage/output/dyad_{dyad}_{iter}_kgi{kgi}_bias{bias}.parquet')
