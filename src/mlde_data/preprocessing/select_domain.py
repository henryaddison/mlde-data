import logging

import xarray as xr

logger = logging.getLogger(__name__)

class SelectDomain:

    # expected index ranges for LONDON_IN_CPM {"grid_latitude": range(142, 196), "grid_longitude": range(337, 391)}
    LONDON_IN_CPM = {"grid_latitude": slice(-1.51995003, -0.45995), "grid_longitude": slice(361.03076172, 362.09075928)}

    LONDON_IN_CPM_64x64 = dict(grid_longitude=slice(360.86076, 362.13074), grid_latitude=slice(-1.57995, -0.31995))

    # expected index ranges for LONDON_IN_GCM {"projection_x_coordinate": [12, 13], "projection_y_coordinate": [4, 5]}
    LONDON_IN_GCM = {"projection_x_coordinate": slice(510000., 570000.), "projection_y_coordinate": slice(150000., 210000.)}

    def __init__(self, subdomain_defn) -> None:
        self.subdomain_defn = subdomain_defn

    def run(self, ds):
        logger.info(f"Selecting subdomain {self.subdomain_defn}")

        return ds.sel(self.subdomain_defn)
