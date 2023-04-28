from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import geopandas as gpd

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.preprocessing import StandardScaler
from matplotlib.axes import Axes
from matplotlib.cm import coolwarm

import gpflow
from gpflow.utilities import print_summary
from typing import Union, Tuple

pd.options.mode.chained_assignment = None

class KGIgpp:
    """
    A Gaussian Point Process fitting and prediction model for conflict models.
    Uses latent actor/dyad information to fit a monthly actor-based GPP model.
    Applies a user defined decay function for older data.
    Trains on GPU if available.
    """
    def __init__(self,
                 X: Union[np.array,pd.DataFrame], 
                 y: Union[np.array,pd.DataFrame],
                 kernel: gpflow.kernels.Kernel) -> None:
        """
        X: a 2-D numpy-array or pandas data frame containing lat/lon, x/y coordinates for observed points. 
        y: a vector (numpy-array) of y coordinates (deaths measured at points) 
        kernel: A GPFlow tensor-flow based kernel
        """
        if type(X) == pd.DataFrame:
            X = X.to_numpy()
        if type(y) == pd.DataFrame:
            y = y.to_numpy()
        self.X = X
        self.y = y
        self.kernel = kernel
        self.model = None
        self.Xplot = None
        self.f_mean = None

    def dimensional_bbox(self, dim = 0, step = 0.1) -> Tuple[Tuple[float,float],float]:
        """
        Returns the bounding box and the number of steps of a series of 2D observed points
        This is done in 1D space (i.e. lat space or lon space), with a 15-step buffer (1.5 WGS units).
        E.g. for three points at lat 1.5, 1.25 and 2.25, the buffer will be 1.5-1.5 = 0 and 2.25+1.5 = 3.75
        dim : The dimension of the 2-D array (0 or 1)
        step : The size in wgs units, for the step you want to use.
        returns : a tuple containings: ((min_coord_1d, max_coord_1d), number_of_steps)
        """
        extents = np.min(self.X[:, dim])-15*step, np.max(self.X[:, dim])+15*step
        steps = int((extents[1]-extents[0])/step)
        return extents, steps
    
    @staticmethod
    def pad_extents(extents0, steps0, extents1, steps1):
        """
        Given a non-square bounding box in x/y space defined by two pairs of extents and steps, one for each coordinate
        Make a square by expanding the shorter edge and compute the correct number of steps.
        returns : two new pairs of extents and steps.
        """
        step = (abs(extents1[1]-extents1[0]))/steps1
        pad_size = abs(int((steps1-steps0)/2))

        which_pad = extents0 if steps0<steps1 else extents1
        steps = steps1 if steps0<steps1 else steps0

        new_extent = which_pad[0] - pad_size*step,  which_pad[1] + pad_size*step

        if steps0<steps1:
            return new_extent, steps, extents1, steps
        else:
            return extents0, steps, new_extent, steps
    
    
    def train(self) -> None:
        """
        Train a GPP model and optimize all the parameters (incl. maximum likelihood).
        """
        self.model = gpflow.models.GPR(
            (self.X, self.y), kernel=deepcopy(self.kernel), noise_variance=1e-3
        )    
        gpflow.set_trainable(self.model.likelihood, True)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.model.training_loss, self.model.trainable_variables)
        
        
    def predict_mesh (self, step=0.1, pad_to_square=True) -> Tuple[np.array,np.array,np.array,np.array]:
        """
        Predict a mesh defined by the bounding box of the training data and a step-size (default to 0.1 WGS units, apx. 11 km).
        The mesh represents a probabilistic conflict zone given xy coords.
        Prediction is based on maximum likelihood value if no prior, or on median if prior is supplied.
        step : granularity of the mesh. .1 default, corresponds to .1 WGS84 units, apx. 11 km 
        pad_to_square : make the mesh square by expanding the shorter dimension.
        """
        
        if self.model is None:
            self.train()
        
        extents0, steps0 = self.dimensional_bbox(0,step=step)
        extents1, steps1 = self.dimensional_bbox(1,step=step)

        if pad_to_square:
            extents0, steps0, extents1, steps1 = self.pad_extents(extents0, steps0, extents1, steps1)

        Xplots0 = np.linspace(*extents0, steps0)
        Xplots1 = np.linspace(*extents1, steps1)

        Xplot1, Xplot2 = np.meshgrid(Xplots0, Xplots1)
        Xplot = np.stack([Xplot1, Xplot2], axis=-1)
        Xplot = Xplot.reshape([steps0 * steps1, 2])

        #y_mean, _ = self.model.predict_y(Xplot, full_cov=False)
        f_mean, _ = self.model.predict_f(Xplot, full_cov=False)
        f_mean = f_mean.numpy().reshape((steps1, steps0))
        
        self.Xplot = Xplot
        self.f_mean = f_mean
        
        return Xplot1, Xplot2, f_mean, Xplot
    
    def plot_contour(self, 
                     title="Mesh Plot of fitted GPP...") -> Axes:
        """
        Make a matplotlib countour plot of the fitted GPP
        """
        fig = plt.figure()
        Xplot1, Xplot2, f_mean, Xplot = self.predict_mesh()
        plt.contourf(Xplot2, Xplot1, f_mean)
        ax = fig.axes[0]
        ax.set_title(title)
        return fig
    
    def plot_fitted_3d (self, 
                        *, 
                        title="3D surface plot of fitted GPP",
                        step=0.1, pad_to_square=True) -> Axes:
        """
        Make a 3D plot of the fitted GPP
        """
        
        Xplot1, Xplot2, f_mean, Xplot = self.predict_mesh()
        
        fig = plt.figure()
        _, (ax) = plt.subplots(nrows=1, ncols=1, subplot_kw={"projection": "3d"})
        
        ax.plot_surface(Xplot1, Xplot2, f_mean, cmap=coolwarm, alpha=0.7)
        ax.set_title(title)

        ax.scatter(self.X[:, 0], self.X[:, 1], self.y[:, 0], s=50, c="black")
        ax.set_title(title)
        return fig

    def predict_df(self) -> pd.DataFrame:
        """
        Predict the same mesh as defined in predict_mesh (for the bounding box of the training data + buffer) but in dataframe format.
        """
        Xplot1, Xplot2, f_mean, Xplot = self.predict_mesh()
        preds = pd.DataFrame({'lat':Xplot[:,0],'lon':Xplot[:,1], 'f_mean':f_mean.flatten()})
        return preds
    
    def predict_posterior_df(self, n=1000, alpha=0.05, seed=None) -> pd.DataFrame:
        """
        Predicts same as df but including uncertainty metrics by sampling from the posterior distribution.
        Make predictions with uncertainty by sampling n posterior samples from the posterior sample distribution.
        Computes statistics using credible (conf.) intervals at a conventional alpha value (.05 default) using the percentile method
        Returns aggregate values from the posterior distribution
        """
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
        preds = self.predict_df()
        samps_10 = self.model.predict_f_samples(self.Xplot, n)
        preds['other_f_mean'] = pd.DataFrame(tf.math.reduce_mean(samps_10,axis=0))
        preds['other_f_sd'] = pd.DataFrame(tf.math.reduce_std(samps_10,axis=0))
        preds['other_f_median'] = pd.DataFrame(tfp.stats.percentile(samps_10,q=0.5,axis=0))
        preds['other_f_ci_low'] = pd.DataFrame(tfp.stats.percentile(samps_10,q=alpha/2,axis=0))
        preds['other_f_ci_high'] = pd.DataFrame(tfp.stats.percentile(samps_10,q=1-alpha/2,axis=0))
        return preds
    
    def predict_gdf(self) -> pd.DataFrame:
        """
        Predict the same mesh as defined in predict_mesh (for the bounding box of the training data + buffer) but in geodataframe format.
        """
        preds = self.predict_df()
        return gpd.GeoDataFrame(preds.f_mean,geometry=gpd.points_from_xy(preds.lon,preds.lat,crs=4326))
    
    def predict_posterior_gdf(self, n=1000, alpha=0.05, seed=None) -> pd.DataFrame:
        """
        Predict the same mesh as defined in predict_posterior_df (for the bounding box of the training data + buffer) but in geodataframe format.
        """
        preds = self.predict_posterior_df(n=n, alpha=alpha, seed=None)
        return gpd.GeoDataFrame(preds,geometry=gpd.points_from_xy(preds.lon,preds.lat,crs=4326))

    

def biased_sampling(
    df: pd.DataFrame,
    weighing_var: Union[str,None] = 'log_ttime',
    weighing_frac: float = .2,
    mc_sample: float = .2,
    where_prec_thresh: float = 4,
    seed: Union[int,None] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Implement partially biased sampling.
    :weighing_var : Name of the variable used for sampling bias [0,1]; a higher value will make it more probable that it will be included in the KGI set.
    :weighing_frac : How much should the sampling bias field matter [0,1] with 0 : random_uniform? 
    :mc_sample : How much of the complete sample should be transformed into KGI?
    :where_prec_thresh : What's a complete data value?
    :seed : seed for numpy and tf.
    """
    
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    mc_ged = df.copy(deep=True)
    mc_ged = mc_ged[mc_ged.where_prec <= where_prec_thresh]
    
    if weighing_var is None:
        weighing_var = '__w'
        mc_ged[weighing_var] = 1
        
    mc_ged[weighing_var] = 1-weighing_frac + weighing_frac*mc_ged[weighing_var]
    
    mc_kgi = mc_ged.sample(frac=mc_sample, weights = mc_ged[weighing_var]) 
    mc_train = mc_ged.drop(mc_kgi.index)
    
    return mc_train, mc_kgi


def decay_ged(df: pd.DataFrame, *, lambda_param: float = -0.173286):
    """
    A simple decay function to implement lossy memory for GED time series data.
    Default half life correspons to 5 time-units.
    """
    return df.best*(np.e**(lambda_param*(df.month_id.max()-df.month_id)))


def ged_from_file(id: int, 
                  month_id: int, *, 
                  dyad_level: bool = True, 
                  month_window: int = 24, 
                  version: str = '22.1') -> pd.DataFrame:
    
    ged = pd.read_parquet(f'ged_{version}.parquet')
    
    if dyad_level:
        column = 'dyad_new_id'
    else:
        column = 'conflict_new_id'
    
    filter_ged = ged.query(f'{column} == {int(id)}')
    #return filter_ged
    
    if month_window is None:
        start = 0
    else:
        start = month_id-month_window
        start = start if start>=0 else 0 
    
    #print(start,month_id)
    filter_ged = filter_ged[filter_ged.month_id.between(start, month_id)]
    return filter_ged.reset_index()