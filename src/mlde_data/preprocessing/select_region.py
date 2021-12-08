import logging

import xarray as xr

logger = logging.getLogger(__name__)

class SelectRegion:

    # expected index ranges for LONDON_IN_CPM {"grid_latitude": range(142, 196), "grid_longitude": range(337, 391)}
    LONDON_IN_CPM = {"grid_latitude": slice(-1.51995003, -0.45995), "grid_longitude": slice(361.03076172, 362.09075928)}

    # expected index ranges for LONDON_IN_GCM {"projection_x_coordinate": [12, 13], "projection_y_coordinate": [4, 5]}
    LONDON_IN_GCM = {"projection_x_coordinate": slice(510000., 570000.), "projection_y_coordinate": slice(150000., 210000.)}

    def __init__(self, input_filepath, output_filepath, subregion_defn) -> None:
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.subregion_defn = subregion_defn

    def run(self):
            logger.info(f"Selecting region of {self.input_filepath}")
            da = xr.load_dataset(self.input_filepath)

            subregion_da = da.sel(self.subregion_defn)
            subregion_da.to_netcdf(self.output_filepath)
