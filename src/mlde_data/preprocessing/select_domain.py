import logging
import math

import numpy as np

from ml_downscaling_emulator.utils import cp_model_rotated_pole, platecarree

logger = logging.getLogger(__name__)

class SelectDomain:

    LONDON_LONG_LAT  = (-0.118092, 51.509865)
    LONDON_RP_LONG_LAT = cp_model_rotated_pole.transform_point(*LONDON_LONG_LAT, src_crs=platecarree)

    # expected index ranges for LONDON_IN_GCM {"projection_x_coordinate": [12, 13], "projection_y_coordinate": [4, 5]}
    LONDON_IN_GCM = {"projection_x_coordinate": slice(510000., 570000.), "projection_y_coordinate": slice(150000., 210000.)}

    def __init__(self, subdomain, size=64) -> None:
        self.subdomain = subdomain
        self.size = size

    def run(self, ds):
        logger.info(f"Selecting subdomain {self.subdomain}")
        if self.subdomain == "london":
            centre_ds = ds.sel(grid_longitude=360.0+self.LONDON_RP_LONG_LAT[0], grid_latitude=self.LONDON_RP_LONG_LAT[1], method="nearest")

            centre_long_idx = np.where(ds.grid_longitude.values == centre_ds.grid_longitude.values)[0].item()
            centre_lat_idx = np.where(ds.grid_latitude.values == centre_ds.grid_latitude.values)[0].item()

            radius = self.size - 1
            left_length = math.floor(radius/2.0)
            right_length = math.ceil(radius/2.0)
            down_length = math.floor(radius/2.0)
            up_length = math.ceil(radius/2.0)

            ds = ds.sel(grid_longitude=slice(ds.grid_longitude[centre_long_idx-left_length].values, ds.grid_longitude[centre_long_idx+right_length].values), grid_latitude=slice(ds.grid_latitude[centre_lat_idx-down_length].values, ds.grid_latitude[centre_lat_idx+up_length].values))

            return ds
