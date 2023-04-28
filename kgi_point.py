from copy import deepcopy

import logging

import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import geopandas as gpd

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from shapely.geometry import Polygon, Point

from sklearn.preprocessing import StandardScaler
from matplotlib.axes import Axes
from matplotlib.cm import coolwarm

import gpflow
from gpflow.utilities import print_summary
#from typing import Union

from kgi import KGIgpp, decay_ged, ged_from_file

pd.options.mode.chained_assignment = None

#
# Caching to disk is disabled. The GPP runs on GPU and is fitted more efficiently than storing large objects to disk and retrieving them back
# Time savings are on the orders of 25% wall hours.
# If you fit on slow CPUs and fast storage, may be efficient to turn caching back on.
#
 
#from diskcache import Cache
#cache = Cache('storage/caches/cache')

class ADMShapes:
    """
    Simple class for loading and managing Geopandas administrative units.
    Built for future-proofing with more complex administrative units files.
    """
    def __init__(self,path):
        self.data = gpd.read_file(path)
        
class KGIPoint: 
    """
    A general purpose class for working with KGI points (known geographic imprecision) in conflict (GED/ACLED/ICEWS/SCAD) battle event data
    Handles two basic processes : 
    1. Mapping a KGI point to the shape (based on precision score) using GADM, based on the where_prec conf
    2. Using a fitted ptobabilistic conflict zone Gaussian Point Process intersect the representation of the point
    3. Generate a representation of the conflict on a .1 WGS unit mesh (~10-11 km at the equator)
    4. Sample m simple or multiple imputations of likely locations from this mesh.
    
    As a side effect, you can make a real point into a KGI point by making a KGI point

    """
    def __repr__(self):
        return f"KGIPoint(lat={self.lat}, lon={self.lon}, where_prec={self.where_prec})"
    
    def __str__(self):
        return self.__repr__()
    
    def __init__ (self,
                  lat: float,
                  lon: float,
                  where_prec: int = 4,
                  adm: ADMShapes = None) -> None:
        """
        lat : latitude of KGI point;
        lon : longitude of KGI point;
        where_prec : coordinate precision uses the conventions in GED, 3 for ADM2, 4 for ADM1, 6 for country
        adm : An ADMShapes representation of the administrative/country shapes. In none, where_prec will determine default shapes locations.
        returns : None
        """
        if where_prec not in (3,4,6):
            raise ValueError("Not a fuzzy point!")
        if where_prec == 3:
            raise NotImplementedError("Not yet implemented for prec 3")
        self.where_prec = where_prec
        
        if not(-90<lat<90):
            raise ValueError("Not a valid WGS84 latitude!")
        if not(-180<lon<180):
            raise ValueError("Not a valid WGS84 longitude!")
            
        self.lat = lat
        self.lon = lon
        self.where_prec = where_prec
        self.shapes = ADMShapes('storage/afr_adm1.gpkg') if adm is None and where_prec==4 else adm
        self.shapes = ADMShapes('storage/afr_cntry.gpkg') if adm is None and where_prec==6 else self.shapes

    #@cache.memoize(typed=True, expire=None, tag="eshape")
    @staticmethod
    def __extract_shape(lat: float,
                        lon: float,
                        shp_gdf: gpd.GeoDataFrame,
                        level:int = 4,
                        allowed_buffer:float = 0.1) -> Polygon:
        """
        Extracts a shapely polygon for a given lat/lon WGS84 coordinate points.
        """
        shapes = shp_gdf.copy()
        polygon = shapes[shapes.intersects(Point(lon,lat))].reset_index()
        
        if polygon.shape[0]>=1:
            return polygon.geometry[0]
        
        shapes['cdist'] = shapes.distance(Point(lon,lat))
        shapes = shapes[shapes.cdist<allowed_buffer]
        
        if shapes.shape[0] > 0:
            polygon = shapes.sort_values(by='cdist').reset_index().geometry[0]
        else:
            polygon = None
        return polygon
    
    
    #@cache.memoize(typed=True, expire=None, tag="shape")
    @property
    def polygon(self)->Polygon:
        """
        Get the polygon representation for a KGI point.
        returns : The ADM polygon corresponding to a KGI point.
        """
        return self.__extract_shape(lat=self.lat,
                                    lon=self.lon,
                                    shp_gdf=self.shapes.data,
                                    level=self.where_prec)

    def empty_mesh(self) -> gpd.GeoDataFrame:
        """
        Create an empty mesh (fishnet) corresponding to the bounding box of the KGI polygon
        Mesh size is .1 decimal degree in WGS84.
        returns : .1 x .1 dec degrees (WGS84) empty mesh containing the lat and lon of the centroid of the mesh points and a f_mean column filled with an uniform 0.5 for sampling
        """
        bounds = self.polygon.bounds
        mesh_x = np.linspace(bounds[0],bounds[2], int((bounds[2]-bounds[0])/0.1))
        mesh_y = np.linspace(bounds[1],bounds[3], int((bounds[3]-bounds[1])/0.1))
        mesh_x, mesh_y = np.meshgrid(mesh_x, mesh_y)
        mesh = np.stack([mesh_x, mesh_y], axis=-1)
        mesh = mesh.reshape(mesh.shape[0]*mesh.shape[1],2)
        df = pd.DataFrame({'lat':mesh[:,1],'lon':mesh[:,0], 'f_mean':0.5})
        return gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.lon,df.lat,crs=4326))
            
    #@cache.memoize(typed=True, expire=None, tag="2dsampling")
    def intersect_zone(self, 
                       kgi: KGIgpp) -> gpd.GeoDataFrame:
        """
        Creates a clipped mesh (fishnet) to the shape of the polygon
        returns : returns a clipped GeoDataFrame containing only the clipped mesh.
        """
        if kgi is None or kgi.X.shape[0] == 0:
            logging.warning("No training data or no KGI conflict mesh! Proceeding without training (doing random assignment).")
            dfx = self.empty_mesh()
        else:    
            #print('Right')
            dfx = kgi.predict_gdf()
        #print(self.polygon)
        poly = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[self.polygon])
        sampling_run = gpd.sjoin(dfx, poly, how='inner', predicate='intersects')
        try:
            del sampling_run['index_right']
        except:
            pass
        return sampling_run.fillna(0)
    
    def sample_gdf(self,
              kgi: KGIgpp,
              n: int = 100,
              bias: float = 0,
              seed: int = None,
              as_centroids = False) -> gpd.GeoDataFrame:
        """
        Sample n samples of likely locations of based on a fitted probabilistic gaussian point process model.
        kgi : A fitted KGIgpp object containing a probabilistic conflict zone, fitted (trained).
        n : Number of samples (probable points for the KGI point to sample). 
        bias : Add a bias to the weights used to sample data.
        seed : Probabilistic seed. If None, the seed is random.
        returns : A pandas data frame containing the n sampled points and their associated f_mean value for their point
        """
        
        if seed is not None:
            np.random.seed(seed)

        dfx = self.intersect_zone(kgi).copy()
        dfx = dfx[dfx.f_mean>0].reset_index(drop=True)
        
        if dfx.shape[0]==0:
            dfx = self.intersect_zone(None).reset_index()
            logging.warning(f"No conflict zone intersection found for point {self}. Proceeding with random assignment!")
        
        dfx = dfx.sample(n=n, weights='f_mean', replace=True, ignore_index=True).reset_index(drop=True)
        return dfx