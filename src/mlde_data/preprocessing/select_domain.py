import logging

import numpy as np

from ml_downscaling_emulator.utils import cp_model_rotated_pole, platecarree

logger = logging.getLogger(__name__)

class SelectDomain:

    # expected index ranges for LONDON_IN_CPM {"grid_latitude": range(142, 196), "grid_longitude": range(337, 391)}
    LONDON_IN_CPM = {"grid_latitude": slice(-1.51995003, -0.45995), "grid_longitude": slice(361.03076172, 362.09075928)}

    LONDON_IN_CPM_64x64 = dict(grid_longitude=slice(360.86076, 362.13074), grid_latitude=slice(-1.57995, -0.31995))

    LONDON_LONG_LAT  = (-0.118092, 51.509865)
    LONDON_RP_LONG_LAT = cp_model_rotated_pole.transform_point(*LONDON_LONG_LAT, src_crs=platecarree)

    # expected index ranges for LONDON_IN_GCM {"projection_x_coordinate": [12, 13], "projection_y_coordinate": [4, 5]}
    LONDON_IN_GCM = {"projection_x_coordinate": slice(510000., 570000.), "projection_y_coordinate": slice(150000., 210000.)}

    def __init__(self, subdomain) -> None:
        self.subdomain = subdomain

    def run(self, ds):
        logger.info(f"Selecting subdomain {self.subdomain}")
        if self.subdomain == "london":
            centre_ds = ds.sel(grid_longitude=360.0+self.LONDON_RP_LONG_LAT[0], grid_latitude=self.LONDON_RP_LONG_LAT[1], method="nearest")

            centre_long_idx = np.where(ds.grid_longitude.values == centre_ds.grid_longitude.values)[0].item()
            centre_lat_idx = np.where(ds.grid_latitude.values == centre_ds.grid_latitude.values)[0].item()

            ds = ds.sel(grid_longitude=slice(ds.grid_longitude[centre_long_idx-31].values, ds.grid_longitude[centre_long_idx+32].values), grid_latitude=slice(ds.grid_latitude[centre_lat_idx-31].values, ds.grid_latitude[centre_lat_idx+32].values))

            return ds
            # london_cp_rp_long_lat_idx = (ds.coord('grid_longitude').nearest_neighbour_index(london_rp_long_lat[0]+360), ds.coord('grid_latitude').nearest_neighbour_index(london_rp_long_lat[1]))

        # return ds.sel(self.subdomain_defn)
