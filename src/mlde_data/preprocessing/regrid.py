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

    def __init__(self, orig_ds, variable, scheme="nn") -> None:
        self.orig_ds = orig_ds
        self.variable = variable
        self.scheme = self.SCHEMES[scheme]()

        pass

    def run(self, ds):

        # regrid the coarsened data to match the original horizontal grid (using NN interpolation)
        # NB iris and xarray can only comminicate in dataarrays not datasets
        # and form a dataset based on the original hi-res with this new coarsened then NN-gridded data
        # target_cube = iris.load_cube(self.target_grid_filepath)

        regridder = self.scheme.regridder(ds[self.variable].to_iris(), self.orig_ds['xwind'].to_iris())
        regridded_da = regridder(ds[self.variable].to_iris())

        # TODO: better way to create a dataset from the regridded DataArray
        self.orig_ds[self.variable] = xr.DataArray.from_iris(regridded_da)

        return self.orig_ds
