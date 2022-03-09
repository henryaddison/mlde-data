import logging

import xarray as xr

logger = logging.getLogger(__name__)

class Constrain:
    def __init__(query):
        self.query = query

    def run(self, ds):
        logger.info(f"Selecting {query} portion of dataset")
        return ds.sel(query)
