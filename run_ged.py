import warnings

import click
import pandas as pd
import numpy as np
import tensorflow as tf
import gpflow
from mingester.extensions import ViewsMonth as lmonth
from gpflow.utilities import print_summary
from typing import Union
from kgi import KGIgpp, decay_ged, ged_from_file
from kgi_point import ADMShapes, KGIPoint
from shapely.geometry import Polygon, Point
from functools import partial
import tensorflow_probability as tfp

from data_manager import ged_consolidated_from_disk_or_api


def process_row(row: pd.DataFrame, kgi1: KGIgpp, kgi2: KGIgpp, level, shape) -> pd.DataFrame:
    #print(row)
    #print(shape)
    kgi_point = KGIPoint(row.latitude,row.longitude,level,shape)
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

def make_run(row, hard_limit=2500):
    print (row.dyad_new_id, row.month_id)
    for m_window in reversed(list(range(3, 24 + 1))):
        train_d = train[(train.dyad_new_id == row.dyad_new_id) & (train.month_id.between(row.month_id-m_window,
                                                                                         row.month_id))]
        if train_d.shape[0] < hard_limit or m_window == 3:
            if m_window < 24:
                warnings.warn(f"""Warning! Hard limit of {hard_limit} points exceeded! 
                              Reduced temporal window to {m_window} months instead of 24 months.
                              This yields {train_d.shape[0]} points!""")
            break

    train_d['best']=decay_ged(train_d)
    train_d = train_d[['latitude','longitude','best']].groupby(['latitude','longitude']).sum()
    train_d = np.log1p(train_d)

    train_d = train_d[train_d.best>0].reset_index()
    train_d['best'] = train_d['best'].astype('float')

    #print(train_d); exit(0)

    test_4 = ged_test_4[(ged_test_4.dyad_new_id==row.dyad_new_id) & (ged_test_4.month_id == row.month_id)]
    test_6 = ged_test_6[(ged_test_6.dyad_new_id==row.dyad_new_id) & (ged_test_6.month_id == row.month_id)]

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

    out4 = pd.DataFrame()
    if test_4.shape[0] > 0:
        process_this = partial(process_row, kgi1=kgi1, kgi2=kgi2, level=4, shape=adm1)
        out4 = test_4.apply(process_this, axis=1)
        out4 = pd.concat([*out4],ignore_index=True)
        out4['id'] = out4['id'].astype('int')

    out6 = pd.DataFrame()
    if test_6.shape[0] > 0:
        process_this = partial(process_row, kgi1=kgi1, kgi2=kgi2, level=6, shape=cshp)
        out6 = test_6.apply(process_this, axis=1)
        out6 = pd.concat([*out6],ignore_index=True)
        out6['id'] = out6['id'].astype('int')

    done_out = pd.concat([out4, out6], ignore_index=True)
    #print(f"Rows: {done_out.shape[0]}")
    return done_out




@click.command()
@click.option('--seed',default=42,help='MonteCarlo Iteration ID.')
@click.option('--dyad',default=None, help='Dyad')
@click.option('--month_id',default=None, help='Month ID to process')
@click.option('--rebuild', is_flag=True, default=False, help='Rebuild')

def read_opt(seed, dyad, month_id, rebuild):
    """Read options for training KGI models across MC simulations."""
    return seed, dyad, month_id, rebuild


if __name__ == '__main__':
    seed, dyad, month_id, rebuild = read_opt(standalone_mode=False)
    print(f"{seed=}, {dyad=}, {month_id=}, {rebuild=}")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    adm1 = ADMShapes("storage/adm1.gpkg")
    cshp = ADMShapes("storage/cntry.gpkg")

    if month_id is None:
        month_id = lmonth.now().id
    else:
        month_id = int(month_id)

    ged = ged_consolidated_from_disk_or_api(month_id=month_id).rename(columns={'pg_id': 'priogrid_gid'})
    ged = ged[['id',
               'dyad_new_id',
               'best',
               'latitude',
               'longitude',
               'month_id',
               'where_prec',
               'priogrid_gid']]

    train = ged[ged.where_prec <= 3]
    test_q = ged[ged.where_prec.isin([4, 6])]
    ged_test_4 = ged[ged.where_prec == 4]
    ged_test_6 = ged[ged.where_prec == 6]


    dyad_todo_list = test_q[['dyad_new_id',
                             'month_id']].drop_duplicates().reset_index(drop=True).\
        sort_values(by=['dyad_new_id', 'month_id'])

    test_q = None

    if dyad is not None:
        dyad = int(dyad)
        print (f"Only doing dyads == {dyad}")
        dyad_todo_list = dyad_todo_list[dyad_todo_list.dyad_new_id == dyad]

    if rebuild is False:
        print(f"Only doing months == {month_id}")
        dyad_todo_list = dyad_todo_list[dyad_todo_list.month_id == month_id]

    #exit(0)


    train = ged[ged.where_prec <= 3]
    ged_test_4 = ged[ged.where_prec == 4]
    ged_test_6 = ged[ged.where_prec == 6]
    adm1 = ADMShapes("storage/adm1.gpkg")
    cshp = ADMShapes("storage/cntry.gpkg")

    print(f"To do: {dyad_todo_list.shape[0]} dyad months")

    for dyad in dyad_todo_list.dyad_new_id.unique():
        print(f'Now doing {dyad=}')
        run_list = dyad_todo_list[dyad_todo_list.dyad_new_id == dyad]
        out = run_list.apply(make_run, axis=1)
        out_dat = pd.concat([*out], ignore_index=True)
        if rebuild:
            out_dat.to_parquet(f'storage/output/ged_real_dyad_{dyad}_rebuild.parquet')
        else:
            out_dat.to_parquet(f'storage/output/ged_real_dyad_{dyad}_{month_id}.parquet')



