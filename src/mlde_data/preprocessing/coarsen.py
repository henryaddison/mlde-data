import logging

import iris.analysis
import xarray as xr

logger = logging.getLogger(__name__)

class Coarsen:
    def __init__(self, scale_factor, variable):
        self.scale_factor = scale_factor
        self.variable = variable

    def run(self, hi_res_ds):
        logger.info(f"Coarsening {self.variable} by a scale factor of {self.scale_factor}")
        # hi_res_ds = xr.load_dataset(self.input_filepath)

        # horizontally coarsen the hi resolution data
        lo_res_ds = hi_res_ds.coarsen(grid_latitude=self.scale_factor, grid_longitude=self.scale_factor, boundary="trim").mean()

        # return lo_res_ds

        # regrid the coarsened data to match the original horizontal grid (using NN interpolation)
        # NB iris and xarray can only comminicate in dataarrays not datasets
        regridder = iris.analysis.Nearest().regridder(lo_res_ds[self.variable].to_iris(), hi_res_ds[self.variable].to_iris())
        regridded_coarse_da = regridder(lo_res_ds[self.variable].to_iris())
        # and form a dataset based on the original hi-res with this new coarsened then NN-gridded data
        hi_res_ds[self.variable] = xr.DataArray.from_iris(regridded_coarse_da)

        return hi_res_ds
