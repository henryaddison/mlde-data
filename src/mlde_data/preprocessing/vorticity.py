import metpy.calc as mpcalc
import xarray as xr

class Vorticity:
    def __init__(self):
        pass

    def run(self, ds):
        dx, dy = mpcalc.lat_lon_grid_deltas(ds.grid_longitude.values, ds.grid_latitude.values)

        ds['vorticity'] = mpcalc.vorticity(ds.isel(time=10)['xwind'], ds.isel(time=10)['ywind'], dx=dx, dy=dy)

        ds = ds.drop_vars(['xwind', 'ywind'])

        # units='s**-1',
        # standard_name='atmosphere_relative_vorticity',
        # long_name='relative_vorticity'

        return ds
