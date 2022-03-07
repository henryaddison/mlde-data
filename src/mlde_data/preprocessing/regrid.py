import glob
import os

import iris
import xarray as xr

"""
Regrid a dataset based on a given target grid file
"""
class Regrid:

    SCHEMES = {
        "linear": iris.analysis.Linear,
        "nn": iris.analysis.Nearest,
        "area-weighted": iris.analysis.AreaWeighted
    }

    def __init__(self, target_grid_filepath, variable, scheme="nn") -> None:
        self.target_grid_filepath = target_grid_filepath
        self.variable = variable
        self.scheme = self.SCHEMES[scheme]()

        pass

    def run(self, ds):

        # regrid the coarsened data to match the original horizontal grid (using NN interpolation)
        # NB iris and xarray can only comminicate in dataarrays not datasets
        # and form a dataset based on the original hi-res with this new coarsened then NN-gridded data
        target_cube = iris.load_cube(self.target_grid_filepath)

        regridder = self.scheme.regridder(ds[self.variable].to_iris(), target_cube)
        regridded_da = regridder(ds[self.variable].to_iris())

        # TODO: this won't match the dimensions anymore, need to clone the target
        ds[self.variable] = xr.DataArray.from_iris(regridded_da)

        return ds
