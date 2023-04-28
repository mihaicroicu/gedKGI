from .Priogrid import Priogrid
from .ViewsMonth import ViewsMonth
from .miniscratch import fetch_ids
import pandas as pd
import warnings
from functools import partial

@pd.api.extensions.register_dataframe_accessor("pg")
class PgAccessor:
    """
    A Pandas (pd) data-frame accessor for the ViEWS priogrid (pg) class.
    Can be explored using df.pg.* and pd.DataFrame.pg.*
    Any dataframe having a pg_id column can be automagically used with this accessor.
    You can also construct one from latitude and longitude.
    """
    def __init__(self, pandas_obj):
        """
        Initializes the accessor and validates that it really is a priogrid df
        :param pandas_obj: A pandas DataFrame containing
        """
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._obj.pg_id = self._obj.pg_id.astype('int')

    @staticmethod
    def _validate(obj):
        """
        :param obj: An Pandas DF containing a pg_id column
        :return: Nothing. Will crash w/ ValueError if invalid, per Pandas documentation
        """
        if "pg_id" not in obj.columns:
            raise AttributeError("Must have a pg_id column!")
        mid = obj.pg_id.copy()
        if mid.dtypes != 'int':
            warnings.warn('pg_id is not an integer - will try to typecast in place!')
            mid = mid.astype('int')
        if mid.min() < 1:
            raise ValueError("Negative pg_id encountered!")
        if mid.max() > 259200:
            raise ValueError("pg_id out of bounds!")

    @property
    def is_unique(self):
        """
        Determines if the dataframe is pg-unique (contains only unique priogrids)
        :return: Will determine if the dataframe conatins unique pg_id values
        """
        if pd.unique(self._obj.pg_id).size == self._obj.pg_id.size:
            return True
        return False


    @property
    def lat(self):
        """
        Computes the latitude of the centroid of each dataframe row, per priogrid definitions.
        :return: A latitude in WGS-84 format (decimal degrees).
        """
        return self._obj.apply(lambda x: Priogrid(x.pg_id).lat, axis=1)

    @property
    def lon(self):
        """
        Computes the longitude of the centroid of each dataframe row, per priogrid definitions.
        :return: A longitude in WGS-84 format (decimal degrees)
        """
        return self._obj.apply(lambda x: Priogrid(x.pg_id).lon, axis=1)


    @property
    def row(self):
        """
        Computes the row of the centroid of each dataframe row, per priogrid definitions.
        :return: A longitude in WGS-84 format (decimal degrees)
        """
        return self._obj.apply(lambda x: Priogrid(x.pg_id).row, axis=1)

    @property
    def col(self):
        """
        Computes the col of the centroid of each dataframe row, per priogrid definitions.
        :return: A longitude in WGS-84 format (decimal degrees)
        """
        return self._obj.apply(lambda x: Priogrid(x.pg_id).col, axis=1)

    def db_id(self):
        return self._obj

    @classmethod
    def from_latlon(cls, df, lat_col='lat', lon_col='lon'):
        """
        Given an arbitrary dataframe containing two columns, on for latitude (lat) and one for longitude (lon)
        Will return a dataframe containing pg_ids in the pg_id column, that can be used with the df.pg accessor.
        Note: Will crash with ValueError if lat/lon is malformed or contain nulls.
        Use self.soft_validate_latlon to soft-validate the dataframe if your input can be malformed.
        :param df: A dataframe containing a latitutde and a longitude column in WGS84 (decimal degrees) format
        :param lat_col: The name of the Latitude column
        :param lon_col: The name of the Longitude column
        :return: A pg-class dataframe.
        """
        z = df.copy()
        if z.shape[0] == 0:
            z['pg_id'] = None
            return z
        z['pg_id'] = df.apply(lambda row: Priogrid.latlon2id(lat=row[lat_col], lon=row[lon_col]), axis=1)
        return z

    @staticmethod
    def __soft_validate_row(row):
        try:
            _ = Priogrid(int(row['pg_id']))
            ok = True
        except:
            ok = False
        return ok

    @classmethod
    def soft_validate(cls, df):
        z = df.copy()
        if z.shape[0] == 0:
            z['valid_id'] = None
            return z
        z['valid_id'] = z.apply(PgAccessor.__soft_validate_row, axis=1)
        return z


    @staticmethod
    def __soft_validate_pg(row, lat_col, lon_col):
        try:
            _ = Priogrid.latlon2id(lat=row[lat_col], lon=row[lon_col])
            ok = True
        except ValueError:
            ok = False
        return ok

    @classmethod
    def soft_validate_latlon(cls, df, lat_col='lat', lon_col='lon'):
        """
        Soft-validate a df containing lat/lon values. Will produce a valid_latlon column to the existing dataframe
        :param df:
        :param lat_col:
        :param lon_col:
        :return:
        """
        z = df.copy()
        if z.shape[0] == 0:
            z['valid_latlon'] = None
            return z
        soft_validator = partial(PgAccessor.__soft_validate_pg, lat_col=lat_col, lon_col=lon_col)
        z['valid_latlon'] = z.apply(soft_validator, axis=1)
        return z

    def full_set(self, land_only=True):
        x = self._obj
        ctrl_grids = set(range(1, 259201))
        if land_only:
            ctrl_grids = set(i for i in fetch_ids('priogrid')[0])
        if set(x.pg_id) == ctrl_grids:
            return True
        return False

    def get_bbox(self, only_views_cells=False):
        test_square = self._obj
        min_row = test_square.pg.row.min()
        max_row = test_square.pg.row.max()
        min_col = test_square.pg.col.min()
        max_col = test_square.pg.col.max()
        square = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                square += [Priogrid.from_row_col(row=row, col=col).id]
        square = set(square)
        if only_views_cells:
            views_cells, _ = fetch_ids('priogrid')
            square = set(views_cells).intersection(square)
        return square

    def is_bbox(self, only_views_cells=False):
        square = self.get_bbox(only_views_cells=only_views_cells)
        return square == set(self._obj.pg_id)

    def fill_bbox(self, fill_value=None):
        extent = pd.DataFrame({'pg_id': list(self.get_bbox())}).merge(self._obj, how='left', on='pg_id')
        if 'pgm_id' in self._obj:
            extent = PGMAccessor.__db_id(extent)
        if fill_value is not None:
            extent = extent.fillna(extent)
        return extent

@pd.api.extensions.register_dataframe_accessor("m")
class MAccessor():
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._obj.month_id = self._obj.month_id.astype('int')

    @staticmethod
    def _validate(obj):
        if "month_id" not in obj.columns:
            raise AttributeError("Must have a month_id column!")
        mid = obj.month_id.copy()
        if mid.dtypes != 'int':
            warnings.warn('month_id is not an integer - will try to typecast in place!')
            mid = mid.astype('int')
        if mid.min() < 1:
            raise ValueError("Negative month_id encountered")

    @property
    def is_unique(self):
        if pd.unique(self._obj.month_id).size == self._obj.month_id.size:
            return True
        return False

    @property
    def year(self):
        return self._obj.apply(lambda x: ViewsMonth(x.month_id).year, axis=1)

    @property
    def month(self):
        return self._obj.apply(lambda x: ViewsMonth(x.month_id).month, axis=1)

    @classmethod
    def from_year_month(cls, df, year_col='year', month_col='month'):
        z = df.copy()
        if z.shape[0] == 0:
            z['month_id'] = None
            return z

        z['month_id'] = z.apply(lambda row: ViewsMonth.from_year_month(year=row[year_col],
                                                                       month=row[month_col]).id,
                                axis=1)
        return z

    @classmethod
    def from_datetime(cls, df, datetime_col='datetime'):
        z = df.copy()
        if z.shape[0] == 0:
            z['month_id'] = None
            return z

        z['temp_year_col'] = z[datetime_col].dt.year
        z['temp_month_col'] = z[datetime_col].dt.month
        z['month_id'] = z.apply(lambda row: ViewsMonth.from_year_month(year=row['temp_year_col'],
                                                                       month=row['temp_month_col']).id,
                                axis=1)
        del z["temp_year_col"]
        del z["temp_month_col"]
        return z

    def db_id(self):
        return self._obj

    def fill_panel_gaps(self):
        extent = pd.DataFrame({'month_id': range(self._obj.month_id.min(), self._obj.month_id.max() + 1)})
        extent = extent.merge(self._obj, how='left', on=['month_id'])
        return extent

    @staticmethod
    def __soft_validate(row):
        try:
            if 0 < row['month_id'] < 1000:
                return True
            else:
                return False
        except:
            return False

    @classmethod
    def soft_validate(cls, df):
        z = df.copy()
        if z.shape[0] == 0:
            z['valid_id'] = None
            return z
        z['valid_id'] = z.apply(MAccessor.__soft_validate, axis=1)
        return z

    @staticmethod
    def __soft_validate_month(row, year_col, month_col):
        try:
            _ = ViewsMonth.from_year_month(year=row[year_col], month=row[month_col]).id
            ok = True
        except ValueError:
            ok = False
        return ok

    @classmethod
    def soft_validate_year_month(cls, df, year_col='year', month_col='month'):
        z = df.copy()
        if z.shape[0] == 0:
            z['valid_year_month'] = None
            return z
        soft_validator = partial(MAccessor.__soft_validate_month, year_col=year_col, month_col=month_col)
        z['valid_year_month'] = z.apply(soft_validator, axis=1)
        return z

    def full_set(self, max_month=None):
        if max_month is not None:
            if set(range(190, max(self._obj.month_id)+1)) != set(range(190, max_month+1)):
                return False
            else:
                return True
        else:
            av_months = set(fetch_ids('month')[0])
            if set(range(190,max(self._obj.month_id)+1)) != av_months:
                return False
            else:
                return True



@pd.api.extensions.register_dataframe_accessor("pgm")
class PGMAccessor(PgAccessor, MAccessor):
    def __init__(self, pandas_obj):
        super().__init__(pandas_obj)

    @property
    def is_unique(self):
        uniques = self._obj[['pg_id', 'month_id']].drop_duplicates().shape[0]
        totals = self._obj.shape[0]
        if uniques == totals:
            return True
        return False

    @classmethod
    def from_datetime_latlon(cls, df, datetime_col='datetime', lat_col='lat', lon_col='lon'):
        z = df.copy()
        z = super().from_datetime(z, datetime_col = datetime_col)
        z = super().from_latlon(z, lat_col=lat_col, lon_col=lon_col)
        return z

    @classmethod
    def from_year_month_latlon(cls, df, year_col='year', month_col='month', lat_col='lat', lon_col='lon'):
        z = df.copy()
        z = super().from_year_month(z, year_col=year_col, month_col=month_col)
        z = super().from_latlon(z, lat_col=lat_col, lon_col=lon_col)
        return z

    @classmethod
    def soft_validate_year_month_latlon(cls, df, year_col='year', month_col='month', lat_col='lat', lon_col='lon'):

        z = df.copy()
        if z.shape[0] == 0:
            z['valid_year_month_latlon'] = None
            return z

        z = super().soft_validate_year_month(z, year_col=year_col, month_col=month_col)
        z = super().soft_validate_latlon(z, lat_col=lat_col, lon_col=lon_col)
        z['valid_year_month_latlon'] = z.valid_year_month & z.valid_latlon
        return z

    @property
    def country(self):
        raise NotImplementedError("""Due to the asymmetric format of the DB (PGM is just Africa and ME), 
        converting between PGM and CM is not supported in ingester as data loss. Use PGY->CY instead!""")

    @classmethod
    def soft_validate(cls, df):
        z = df.copy()
        if z.shape[0] == 0:
            z['valid_id'] = None
            return z
        else:
            z = PgAccessor.soft_validate(z)
            z['valid_id_0'] = z.valid_id
            del z['valid_id']
            z = MAccessor.soft_validate(z)
            z['valid_id_1'] = z.valid_id
            del z['valid_id']
            z['valid_id'] = z.valid_id_1 & z.valid_id_0
            del z['valid_id_0']
            del z['valid_id_1']
            return z


    def full_set(self, land_only=True, max_month=None):
        pg_full_set = super(PgAccessor, self).full_set(land_only)
        m_full_set = MAccessor(self._obj).full_set(max_month)
        return pg_full_set & m_full_set

    def is_panel(self):
        test_square = self._obj
        min_month = test_square.month_id.min()
        max_month = test_square.month_id.max()
        if len(test_square.month_id.unique()) != max_month - min_month + 1:
            return False
        first_panel = set(test_square[test_square.month_id == min_month].pg_id.unique())
        for month in test_square.month_id.unique():
            cur_panel = set(test_square[test_square.month_id == month].pg_id.unique())
            if first_panel != cur_panel:
                return False
        return True

    def is_complete_cross_section(self, only_views_cells=True):
        x = self._obj
        if not self.is_bbox(only_views_cells=only_views_cells):
            return False
        vc_len = len(fetch_ids('priogrid')[0]) if only_views_cells else 259200
        if len(x.pg_id.unique()) == vc_len:
            return True
        return False

    def is_complete_time_series(self, min_month=190, max_month=621):
        x = self._obj
        if not self.is_panel():
            return False
        if x.test_square.month_id.min() != min_month:
            return False
        if x.test_square.month_id.max() != max_month:
            return False
        return True


    def fill_panel_gaps(self, fill_value=None):
        extent1 = pd.DataFrame({'month_id': range(self._obj.month_id.min(), self._obj.month_id.max() + 1), 'key': 0})
        extent2 = pd.DataFrame({'key': 0, 'pg_id': self._obj.pg_id.unique()})
        extent = extent1.merge(extent2, on='key')[['pg_id','month_id']]
        extent = extent.merge(self._obj, how='left', on=['pg_id','month_id'])
        if 'pgm_id' in self._obj:
            extent = PGMAccessor.__db_id(extent)
        if fill_value is not None:
            extent = extent.fillna(fill_value)
        return extent

    def fill_spatial_gaps(self, fill_value=None):
        extent1 = pd.DataFrame({'pg_id': self._obj.pg_id.unique(), 'key': 0})
        extent2 = pd.DataFrame({'key': 0, 'month_id': self._obj.month_id.unique()})
        extent = extent1.merge(extent2, on='key')[['pg_id','month_id']]
        extent = extent.merge(self._obj, how='left', on=['pg_id','month_id'])
        if 'pgm_id' in self._obj:
            extent = PGMAccessor.__db_id(extent)
        if fill_value is not None:
            extent = extent.fillna(fill_value)
        return extent

    def fill_bbox(self, fill_value=None):
        extent1 = pd.DataFrame({'pg_id': list(self.get_bbox()), 'key': 0})
        extent2 = pd.DataFrame({'key': 0, 'month_id': list(self._obj.month_id.unique())})
        extent3 = extent1.merge(extent2, on='key')[['pg_id','month_id']]
        extent = extent3.merge(self._obj, how='left', on=['pg_id','month_id'])
        if 'pgm_id' in self._obj:
            extent = PGMAccessor.__db_id(extent)
        if fill_value is not None:
            extent = extent.fillna(fill_value)
        return extent