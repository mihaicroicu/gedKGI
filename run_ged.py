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
import tensorflow_probability as tfp

from data_manager import ged_consolidated_from_disk_or_api


def process_row(row: pd.DataFrame, kgi1: KGIgpp, kgi2: KGIgpp, level, shape) -> pd.DataFrame:
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

def make_run(row):
    print (row.dyad_new_id, row.month_id)
    train_d = train[(train.dyad_new_id==row.dyad_new_id) & (train.month_id.between(row.month_id-24,row.month_id))]
    train_d['best']=decay_ged(train_d)
    train_d = train_d[['latitude','longitude','best']].groupby(['latitude','longitude']).sum()
    train_d = np.log1p(train_d)

    train_d = train_d[train_d.best>0].reset_index()
    train_d['best'] = train_d['best'].astype('float')

    test_4 = ged_test_4[(ged_test_4.dyad_new_id==row.dyad_new_id) & (ged_test_4.month_id==row.month_id)]
    test_6 = ged_test_6[(ged_test_6.dyad_new_id==row.dyad_new_id) & (ged_test_6.month_id==row.month_id)]


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

    process_this = partial(process_row, kgi1=kgi1, kgi2=kgi2, level=4, shape=adm1)
    out4 = test_4.apply(process_this, axis=1)
    out4 = pd.concat([*out4],ignore_index=True)
    out4['id'] = out4['id'].astype('int')

    process_this = partial(process_row, kgi1=kgi1, kgi2=kgi2, level=6, shape=cshp)
    out6 = test_6.apply(process_this, axis=1)
    out6 = pd.concat([*out6],ignore_index=True)
    out6['id'] = out6['id'].astype('int')

    return out4, out6


if __name__ == "__main__":

    dyad = 828

    adm1 = ADMShapes("storage/afr_adm1.gpkg")
    cshp = ADMShapes("storage/afr_cntry.gpkg")

    ged = ged_consolidated_from_disk_or_api(month_id=520).rename(columns={'pg_id':'priogrid_gid'})
    ged = ged[['id','dyad_new_id','best','latitude','longitude','month_id','where_prec','priogrid_gid']]

    train = ged[ged.where_prec <= 3]
    test_q = ged[ged.where_prec.isin(4,6)]
    ged_test_4 = ged[ged.where_prec == 4]
    ged_test_6 = ged[ged.where_prec == 6]


    dyad_todo_list = test_q[['dyad_new_id','month_id']].drop_duplicates().reset_index(drop=True).\
        sort_values(by=['dyad_new_id','month_id'])

    test_q = None

    if dyad is not None:
        dyad = int(dyad)
        print (f"Only doing dyads larger than : {dyad}")
        dyad_todo_list = dyad_todo_list[dyad_todo_list.dyad_new_id==dyad]

    train = ged[ged.where_prec <= 3]
    ged_test_4 = ged[ged.where_prec == 4]
    ged_test_6 = ged[ged.where_prec == 6]


