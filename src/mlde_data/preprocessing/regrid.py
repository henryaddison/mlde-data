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
        self.target_cube = iris.load_cube(target_grid_filepath)
        self.target_ds = xr.open_dataset(target_grid_filepath)
        self.variable = variable
        self.scheme = self.SCHEMES[scheme]()

        pass

    def run(self, ds):
        # regrid the coarsened data to match the original horizontal grid (using NN interpolation)
        # NB iris and xarray can only comminicate in dataarrays not datasets
        # and form a dataset based on the original hi-res with this new coarsened then NN-gridded data

        if ds[self.variable].attrs['grid_mapping'] == "latitude_longitude":
            src_coord_sys = iris.coord_systems.GeogCS(ds["latitude_longitude"].attrs['earth_radius'])
            src_lat_name = 'latitude'
            src_lon_name = 'longitude'
        elif ds[self.variable].attrs['grid_mapping'] == "rotated_latitude_longitude":
            src_coord_sys = iris.coord_systems.RotatedGeogCS(ds["rotated_latitude_longitude"].attrs['grid_north_pole_latitude'], ds["rotated_latitude_longitude"].attrs['grid_north_pole_longitude'], ellipsoid=iris.coord_systems.GeogCS(ds["rotated_latitude_longitude"].attrs['earth_radius']))
            src_lat_name = 'grid_latitude'
            src_lon_name = 'grid_longitude'

        src_cube = ds[self.variable].to_iris()
        # conversion to iris looses the coordinate system on the lat and long dimensions but iris it needs to do regrid
        src_cube.coords(src_lon_name)[0].coord_system = src_coord_sys
        src_cube.coords(src_lat_name)[0].coord_system = src_coord_sys

        regridder = self.scheme.regridder(src_cube, self.target_cube)
        regridded_da = xr.DataArray.from_iris(regridder(src_cube))
        regridded_var_attrs = ds[self.variable].attrs|{"grid_mapping": self.target_ds[self.target_cube.var_name].attrs["grid_mapping"]}

        # forecast_reference_time depends on the time slice but doesn't affect the grid
        # so update for target grid dataset to match the data being regridded
        self.target_ds['forecast_reference_time'] = ds['forecast_reference_time']

        vars = {self.variable: (['time', 'grid_latitude', 'grid_longitude'], regridded_da.values, regridded_var_attrs)}
        vars.update({f'{key}_bnds': ([key, 'bnds'], self.target_ds[f'{key}_bnds'].values, self.target_ds[f'{key}_bnds'].attrs) for key in ['grid_latitude', 'grid_longitude']})
        vars.update({key: (['time', 'bnds'], ds[key].values, ds[key].attrs, {'units': 'hours since 1970-01-01 00:00:00', 'calendar': '360_day'}) for key in ['time_bnds', 'forecast_period_bnds']})

        coords = dict(ds.coords)
        coords['grid_longitude'] = self.target_ds.coords['grid_longitude']
        coords['grid_latitude'] = self.target_ds.coords['grid_latitude']

        ds = xr.Dataset(vars, coords=coords, attrs=ds.attrs)

        return ds
