import glob
import logging
import os

import xarray as xr

logger = logging.getLogger(__name__)

class SelectSubregion:

    # expected index ranges for LONDON_IN_CPM {"grid_latitude": range(142, 196), "grid_longitude": range(337, 391)}
    LONDON_IN_CPM = {"grid_latitude": slice(-1.51995003, -0.45995), "grid_longitude": slice(361.03076172, 362.09075928)}

    # expected index ranges for LONDON_IN_GCM {"projection_x_coordinate": [12, 13], "projection_y_coordinate": [4, 5]}
    LONDON_IN_GCM = {"projection_x_coordinate": slice(510000., 570000.), "projection_y_coordinate": slice(150000., 210000.)}

    def __init__(self, input_dir, output_dir, subregion_defn) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.subregion_defn = subregion_defn

    def run(self):
        output_files = []
        full_da_files = glob.glob(str(self.input_dir / "*.nc"))

        for full_file in full_da_files:
            logger.info(f"Working on {full_file}")
            da = xr.open_dataset(full_file)

            output_filepath = self.output_dir / f"{os.path.basename(full_file)}"

            subregion_da = da.sel(self.subregion_defn)
            subregion_da.to_netcdf(output_filepath)
            output_files.append(output_filepath)

        logger.info("All done")

        return output_files