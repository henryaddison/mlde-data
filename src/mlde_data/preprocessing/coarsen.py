import glob
import logging
import os

import iris.analysis
import xarray as xr

logger = logging.getLogger(__name__)

class Coarsen:
    def __init__(self, input_dir, output_dir, scale_factor=4, variable='pr'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.scale_factor = scale_factor
        self.variable = variable

    def run(self):
        output_files = []
        input_filepaths = glob.glob(str(self.input_dir / "*.nc"))

        for input_filepath in input_filepaths:
            hi_res_ds = xr.load_dataset(input_filepath)

            # horizontally coarsen the hi resolution data
            lo_res_ds = hi_res_ds.coarsen(grid_latitude=self.scale_factor, grid_longitude=self.scale_factor, boundary="trim").mean()

            # regrid the coarsened data to match the original horizontal grid (using NN interpolation)
            # NB iris and xarray can only comminicate in dataarrays not datasets
            regridder = iris.analysis.Nearest().regridder(lo_res_ds[self.variable].to_iris(), hi_res_ds[self.variable].to_iris())
            regridded_coarse_da = regridder(lo_res_ds.pr.to_iris())
            # and form a dataset based on the original hi-res with this new coarsened then NN-gridded data
            hi_res_ds[self.variable] = xr.DataArray.from_iris(regridded_coarse_da)

            output_filepath = self.output_dir / f"{os.path.basename(input_filepath)}"

            hi_res_ds.to_netcdf(output_filepath)

            output_files.append(output_filepath)

        return output_files
