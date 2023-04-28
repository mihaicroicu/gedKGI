import pandas as pd
import requests
from ingester3.extensions import *
from diskcache import Cache
from pathlib import Path

Path("~/.ged_loader_cache/").mkdir(parents=True, exist_ok=True)
ged_cache = Cache('~/.ged_loader_cache/ged_cached.cache')


class GedLoader:
    def __init__(self, version, verbose=True):
        self.version = version
        self.ged = None
        self.__get_month_id()
        self.verbose_print = print if verbose else lambda *a, **k: None

    def __repr__(self):
        verbose = False if self.verbose_print is None else True
        return f"GedLoader({self.version}, verbose={verbose})"

    def __str__(self):
        loaded = "loaded" if self.ged is not None else "NOT loaded"
        return f"GedLoader with version: {self.version}. Data status: {loaded}"

    def __get_month_id(self):
        """
        If trying to load a GED Candidates dataset (20.0.x) infer what ViEWS MonthID it refers to.
        Return nothing otherwise
        """
        self.min_month = None
        self.max_month = None

        if self.version.count('.') == 2:
            # If this is a Candidates dataset, compute the expected month.
            year_extent = int('20' + self.version.split('.')[0])
            month_extent = int(self.version.split('.')[2])
            self.min_month = ViewsMonth.from_year_month(year_extent, month_extent)
            self.max_month = self.min_month
        else:
            # If this is a Candidates dataset, compute the expected month
            self.min_month = ViewsMonth.from_year_month(1989, 1)
            self.max_month = ViewsMonth.from_year_month(int('20' + self.version.split('.')[0])-1, 12)

    @staticmethod
    @ged_cache.memoize(typed=True, expire=None, tag="sliced_slice")
    def _get_ged_slice(next_page_url, token=None):
        headers = {'x-ucdp-access-token': token}
        r = requests.get(next_page_url, headers=headers)
        output = r.json()
        next_page_url = output['NextPageUrl'] if output['NextPageUrl'] != '' else None
        ged = pd.DataFrame(output['Result'])
        page_count = output['TotalPages']
        return next_page_url, ged, page_count

    @property
    def exists(self):
        exists = True
        next_page_url = f"https://ucdpapi.pcr.uu.se/api/gedevents/{self.version}?pagesize=1&page=0"
        try:
            _, _, _ = self._get_ged_slice(next_page_url=next_page_url, token="48dda3460c347f3b")
        except KeyError:
            exists = False
        return exists

    def fetch_ged(self):
        cur_page = 1
        next_page_url = f"https://ucdpapi.pcr.uu.se/api/gedevents/{self.version}?pagesize=500&page=0"

        df = pd.DataFrame()
        while next_page_url:
            self.verbose_print(next_page_url)
            next_page_url, ged_slice, total_pages = self._get_ged_slice(
                next_page_url=next_page_url, token="48dda3460c347f3b")
            df = pd.concat([df, ged_slice], ignore_index=True)
            self.verbose_print(f"{cur_page} of {total_pages} pages loaded.")
            #cur_page += 1

            if cur_page > total_pages:
                ged_cache.clear(retry=True)
                raise ConnectionError('The UCDP API is misbehaving. Try again later!')
            cur_page += 1

        self.ged = df

    def filter_ged(self):
        self.ged = self.ged[self.ged.priogrid_gid >= 1]
        self.ged.date_end = pd.to_datetime(self.ged.date_end)
        self.ged = pd.DataFrame.pgm.from_datetime(self.ged, 'date_end').rename(columns={'priogrid_gid': 'pg_id',
                                                                                        'type_of_violence': 'tv'})
        self.ged = self.ged[self.ged.tv < 4]
        self.ged = self.ged[self.ged.month_id >= self.min_month.id]
        self.ged = self.ged[self.ged.month_id <= self.max_month.id]

    def load(self):
        self.fetch_ged()
        self.filter_ged()